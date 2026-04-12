import logging
import os
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, List, Optional
import threading

import faiss
import torch
from FlagEmbedding import FlagAutoModel
import numpy as np

# 数据目录：默认 /root/data；Docker 或本地可通过环境变量 HOTPOTQA_DATA_ROOT 覆盖（需与建索引时一致）
HOTPOTQA_DATA_ROOT = Path(os.environ.get("HOTPOTQA_DATA_ROOT", "/root/data")).expanduser().resolve()
HOTPOTQA_INDEX_BIN = HOTPOTQA_DATA_ROOT / "index.bin"
# 检索结果展示用段落文本；需与建索引时的 hpqa_corpus.jsonl 一致
HOTPOTQA_CORPUS_JSONL = HOTPOTQA_DATA_ROOT / "hpqa_corpus.jsonl"
# hpqa_corpus.npy 仅 process_hotpotqa 建库时使用，运行时不需要加载

logger = logging.getLogger(__name__)


def default_hotpotqa_embedding_device() -> str:
    """BGE query 编码设备；为当前进程可见 GPU 的 PyTorch 逻辑序号（如 cuda:4），不是 nvidia-smi 物理编号。"""
    return os.environ.get("HOTPOTQA_EMBEDDING_DEVICE", "cpu").strip() or "cpu"


@dataclass
class Passage:
    pid: int
    title: str
    text: str
    score: float = 0.0


@dataclass
class PassagePool:
    passages: List[Passage] = field(default_factory=list)

    def has_passage(self, pid: int) -> bool:
        return any(p.pid == pid for p in self.passages)

    def add_passage(self, passage: Passage) -> None:
        # 按正文去重；pid 用池内序号（避免多次 search 时 parse 侧 pid 恒为 0~4 导致后续检索全被丢弃）
        if any(p.text == passage.text for p in self.passages):
            return
        pid = len(self.passages)
        self.passages.append(
            Passage(pid=pid, title=passage.title, text=passage.text, score=passage.score)
        )

    @property
    def passage_list(self) -> str:
        if not self.passages:
            return "None"
        lines = []
        for i, p in enumerate(self.passages, start=1):
            snippet = p.text[:512].replace("\n", " ")
            lines.append(f"[{i}] (id={p.pid}) {p.title}: {snippet}")
        return "\n".join(lines)


class HotpotQASearchToolLegacy:
    """
    HotpotQA 本地 FAISS 检索，行为对齐大规模训练验证过的实现：
    `Agent-R1-legacy/agent_r1/tool/tools/search_tool.py`（SearchTool）。

    相对上游的增强（不改变成功路径语义）：
    - 数据路径：`HOTPOTQA_DATA_ROOT` + `index.bin` / `hpqa_corpus.jsonl`
    - 进程内共享 index/corpus/model，避免每轨迹重复加载
    - `_format_results` 对非法 id 做边界检查（legacy 直接索引可能 IndexError）
    - `encode_queries` 若为 torch.Tensor 则 `.cpu().numpy()`，便于非 CPU 设备与 FAISS CPU 索引衔接

    上游验证配置：`FlagAutoModel.from_finetuned(..., devices="cpu")`；生产训练建议 CPU 编码，
    若设 `HOTPOTQA_EMBEDDING_DEVICE=cuda:*` 需保证在同一线程调用（见 hotpotqa_agent_flow）。
    """

    _shared_lock = threading.RLock()
    _shared_key: Optional[str] = None
    _shared_index: Optional[faiss.Index] = None
    _shared_corpus: Optional[list[str]] = None
    _shared_model: Optional[FlagAutoModel] = None

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        query_instruction: str = "Represent this sentence for searching relevant passages: ",
        embedding_devices: Optional[str] = None,
    ) -> None:
        self.data_dir = HOTPOTQA_DATA_ROOT
        self.index_path = HOTPOTQA_INDEX_BIN
        self.corpus_path = HOTPOTQA_CORPUS_JSONL
        self.embedding_model_name = embedding_model_name
        self.query_instruction = query_instruction
        dev = (embedding_devices if embedding_devices is not None else default_hotpotqa_embedding_device()).strip() or "cpu"
        self.embedding_devices = dev

        self._index: Optional[faiss.Index] = None
        self._corpus: list[str] = []
        self._model: Optional[FlagAutoModel] = None
        self._ensure_loaded()

    def __enter__(self):
        self._ensure_loaded()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _ensure_loaded(self) -> None:
        cache_key = f"{HOTPOTQA_DATA_ROOT}|{self.embedding_devices}|{self.embedding_model_name}"
        with self.__class__._shared_lock:
            if (
                self.__class__._shared_key != cache_key
                or self.__class__._shared_index is None
                or self.__class__._shared_corpus is None
                or self.__class__._shared_model is None
            ):
                if not self.index_path.exists():
                    raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
                if not self.corpus_path.exists():
                    raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

                index = faiss.read_index(str(self.index_path))
                corpus: list[str] = []
                with self.corpus_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        title = str(rec.get("title", ""))
                        text = str(rec.get("text", ""))
                        corpus.append(f"{title} {text}".strip())

                logger.info(
                    "HotpotQASearchToolLegacy: loading FlagEmbedding model=%s devices=%s",
                    self.embedding_model_name,
                    self.embedding_devices,
                )
                model = FlagAutoModel.from_finetuned(
                    self.embedding_model_name,
                    query_instruction_for_retrieval=self.query_instruction,
                    devices=self.embedding_devices,
                )
                self.__class__._shared_key = cache_key
                self.__class__._shared_index = index
                self.__class__._shared_corpus = corpus
                self.__class__._shared_model = model

                if int(index.ntotal) != len(corpus):
                    logger.warning(
                        "FAISS index.ntotal (%s) != hpqa_corpus.jsonl rows (%s). "
                        "Ids from search may be out of range and passages will be empty; "
                        "rebuild index.bin with the same jsonl or fix the corpus file.",
                        int(index.ntotal),
                        len(corpus),
                    )

            self._index = self.__class__._shared_index
            self._corpus = self.__class__._shared_corpus or []
            self._model = self.__class__._shared_model

    def close(self) -> None:
        # Keep shared model/index alive for whole training process.
        # This matches legacy behavior where SearchTool is initialized once and reused.
        self._index = self.__class__._shared_index
        self._corpus = self.__class__._shared_corpus or []
        self._model = self.__class__._shared_model

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        try:
            query = str(args["query"])
            embeddings = self._encode_queries([query])
            assert self._index is not None
            _, ids = self._index.search(embeddings, 5)
            result_str = self._format_results(ids[0])
            return {"content": result_str, "success": True}
        except Exception as e:
            return {"content": str(e), "success": False}

    def batch_execute(self, args_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """与 Agent-R1-legacy `SearchTool.batch_execute` 相同：单次 encode + search，失败则每条返回同一 str(e)。"""
        if not args_list:
            return []
        try:
            queries = [str(x["query"]) for x in args_list]
            embeddings = self._encode_queries(queries)
            assert self._index is not None
            _, ids = self._index.search(embeddings, 5)
            results_str = [self._format_results(ids[i]) for i in range(len(ids))]
            return [{"content": result_str, "success": True} for result_str in results_str]
        except Exception as e:
            logger.warning(
                "HotpotQASearchToolLegacy.batch_execute failed (%s queries): %s",
                len(args_list),
                e,
                exc_info=True,
            )
            return [{"content": str(e), "success": False} for _ in args_list]

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        self._ensure_loaded()
        with self.__class__._shared_lock:
            assert self._model is not None
            out = self._model.encode_queries(queries)
        # FAISS CPU Index::search 需要主存 float32 ndarray；BGE 在 GPU 上可能返回 torch.Tensor
        if torch.is_tensor(out):
            out = out.detach().float().cpu().numpy()
        arr = np.asarray(out, dtype=np.float32)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr

    def _format_results(self, results) -> str:
        results_list: list[str] = []
        row_ids = [int(x) for x in np.asarray(results, dtype=np.int64).reshape(-1)]
        for result in row_ids:
            if result < 0 or result >= len(self._corpus):
                continue
            results_list.append(self._corpus[result])
        if (
            not results_list
            and self._corpus
            and row_ids
            and max(row_ids) >= len(self._corpus)
        ):
            logger.warning(
                "FAISS returned ids %s but corpus length is %s; dropping all hits.",
                row_ids[:10],
                len(self._corpus),
            )
        return json.dumps({"results": results_list}, ensure_ascii=False)


def parse_legacy_tool_result(content: str) -> list[Passage]:
    """Parse legacy `{"results":[...]}` tool content into Passage list."""
    passages: list[Passage] = []
    try:
        payload = json.loads(content)
        results = payload.get("results", [])
        for idx, text in enumerate(results):
            text_str = str(text)
            passages.append(Passage(pid=idx, title="", text=text_str, score=0.0))
    except Exception:
        return []
    return passages


def format_history_actions(history_actions: list[tuple[str, str]]) -> str:
    """
    history_actions: list of (action_type, payload_str)
    """
    if not history_actions:
        return "None"

    lines: list[str] = []
    for action, payload in history_actions:
        if action == "search":
            lines.append(f"[Search] {payload}")
        else:
            lines.append(f"[{action}] {payload}")
    return "\n".join(lines)