"""Structured document and entity knowledge store."""

from typing import Any, Dict, List, Optional


class KnowledgeStore:
    """Stores structured domain knowledge with citations and provenance."""

    def __init__(self):
        self._documents: Dict[str, Dict[str, Any]] = {}

    def add_document(self, doc_id: str, title: str, text: str, source: str, metadata: Optional[Dict] = None) -> None:
        self._documents[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "source": source,
            "metadata": metadata or {},
        }

    def search_documents(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        tokens = [t.lower() for t in query.split() if len(t) > 3]
        matches = []
        for doc in self._documents.values():
            content = (doc["title"] + " " + doc["text"]).lower()
            score = sum(1 for t in tokens if t in content) if tokens else 1
            if score > 0 or not tokens:
                matches.append((score, doc))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:limit]]
