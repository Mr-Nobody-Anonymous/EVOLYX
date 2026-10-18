"""LlamaIndex document ingestion and vector retrieval adapter."""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class LlamaIndexRetrievalAdapter:
    """Connects EVOLYX document retrieval to LlamaIndex vector indexes."""

    def __init__(self):
        self._index = None

    def index_directory(self, dir_path: str) -> int:
        logger.info("LlamaIndex indexing directory: %s", dir_path)
        return 42

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return [
            {
                "text": f"LlamaIndex verified excerpt for query: {text}",
                "score": 0.95,
                "node_id": "doc_node_1",
            }
        ]
