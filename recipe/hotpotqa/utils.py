import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from FlagEmbedding import FlagAutoModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


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


class WikiQdrantRetriever:
    """
    Simple wrapper around a local/remote Qdrant + BGE embedding model
    for HotpotQA wiki retrieval.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        collection_name: str = "hpqa_corpus",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        data_dir = repo_root / "data" / "corpus" / "hotpotqa"
        default_db_path = data_dir / "qdrant_db"

        self._db_path = Path(db_path) if db_path is not None else default_db_path
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        self._client: Optional[QdrantClient] = None
        self._model: Optional[FlagAutoModel] = None
        self._lock = threading.RLock()

    def __enter__(self):
        self._ensure_client_and_model()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _ensure_client_and_model(self) -> None:
        with self._lock:
            if self._client is None:
                if self.qdrant_url:
                    self._client = QdrantClient(url=self.qdrant_url)
                else:
                    self._client = QdrantClient(path=str(self._db_path))

            if self._model is None:
                self._model = FlagAutoModel.from_finetuned(self.embedding_model_name)

    def close(self) -> None:
        with self._lock:
            if self._model is not None:
                try:
                    stop_fn = getattr(self._model, "stop_self_pool", None)
                    if callable(stop_fn):
                        stop_fn()
                except Exception as e:
                    print(f"[WikiQdrantRetriever.close] stop_self_pool failed: {e}")
                finally:
                    self._model = None

            self._client = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        title_filter: Optional[str] = None,
    ) -> List[Passage]:
        """
        Blocking search API. Caller should run this in a thread pool if used from async code.
        """
        self._ensure_client_and_model()
        assert self._client is not None
        assert self._model is not None

        query_vec = self._model.encode_queries([query])[0]

        q_filter = None
        if title_filter is not None:
            q_filter = Filter(
                must=[
                    FieldCondition(
                        key="title",
                        match=MatchValue(value=title_filter),
                    )
                ]
            )

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            query_filter=q_filter,
            with_payload=True,
        )

        hits = response.points

        passages: List[Passage] = []
        for hit in hits:
            payload = hit.payload or {}
            title = str(payload.get("title", ""))
            text = str(payload.get("text", ""))
            passages.append(
                Passage(
                    pid=int(hit.id),
                    title=title,
                    text=text,
                    score=float(hit.score or 0.0),
                )
            )
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