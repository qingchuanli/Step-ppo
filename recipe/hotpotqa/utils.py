import time
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, List, Optional
import threading

import faiss
from FlagEmbedding import FlagAutoModel
import numpy as np

# 固定数据目录（与预处理产物一致；不通过 Hydra/yaml 配置）
HOTPOTQA_DATA_ROOT = Path("/root/data")
HOTPOTQA_INDEX_BIN = HOTPOTQA_DATA_ROOT / "index.bin"
# 检索结果展示用段落文本；需与建索引时的 hpqa_corpus.jsonl 一致
HOTPOTQA_CORPUS_JSONL = HOTPOTQA_DATA_ROOT / "hpqa_corpus.jsonl"
# hpqa_corpus.npy 仅 process_hotpotqa 建库时使用，运行时不需要加载


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
        if not self.has_passage(passage.pid):
            self.passages.append(passage)

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
    Legacy-compatible FAISS search tool for HotpotQA.
    Mirrors Agent-R1-legacy search_tool.py behavior:
    - name: search
    - input: {"query": "..."}
    - output: {"content": "<json str>", "success": bool}
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
        max_retries: int = 3,
        retry_backoff_s: float = 0.2,
    ) -> None:
        self.data_dir = HOTPOTQA_DATA_ROOT
        self.index_path = HOTPOTQA_INDEX_BIN
        self.corpus_path = HOTPOTQA_CORPUS_JSONL
        self.embedding_model_name = embedding_model_name
        self.query_instruction = query_instruction
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))

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
        cache_key = str(HOTPOTQA_DATA_ROOT)
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

                model = FlagAutoModel.from_finetuned(
                    self.embedding_model_name,
                    query_instruction_for_retrieval=self.query_instruction,
                    devices="cpu",
                )
                self.__class__._shared_key = cache_key
                self.__class__._shared_index = index
                self.__class__._shared_corpus = corpus
                self.__class__._shared_model = model

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
        if not args_list:
            return []
        for attempt in range(1, self.max_retries + 1):
            try:
                queries = [str(x["query"]) for x in args_list]
                embeddings = self._encode_queries(queries)
                assert self._index is not None
                _, ids = self._index.search(embeddings, 5)
                results_str = [self._format_results(ids[i]) for i in range(len(ids))]
                return [{"content": s, "success": True} for s in results_str]
            except Exception:
                if attempt >= self.max_retries:
                    return [{"content": "batch search failed", "success": False} for _ in args_list]
                time.sleep(self.retry_backoff_s * attempt)

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        self._ensure_loaded()
        with self.__class__._shared_lock:
            assert self._model is not None
            return self._model.encode_queries(queries).astype(np.float32)

    def _format_results(self, results: list[int]) -> str:
        results_list: list[str] = []
        for result in results:
            if result < 0 or result >= len(self._corpus):
                continue
            results_list.append(self._corpus[result])
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