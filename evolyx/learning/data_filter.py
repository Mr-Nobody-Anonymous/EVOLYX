"""Data filtering, deduplication, and PII sanitization for training datasets."""

import hashlib
import json
import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class DataFilter:
    """Prepares clean, verified datasets from raw collected trajectories."""

    def __init__(self, min_token_length: int = 10, max_token_length: int = 4096):
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        self._seen_hashes = set()

    def scrub_pii(self, text: str) -> str:
        # Scrub email
        text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_REDACTED]", text)
        # Scrub API keys
        text = re.sub(r"\b(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,})\b", "[SECRET_REDACTED]", text)
        return text

    def compute_hash(self, prompt: str, completion: str) -> str:
        content = f"{prompt.strip()}|{completion.strip()}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def filter_example(self, raw_example: Dict) -> Tuple[bool, str, Dict]:
        prompt = raw_example.get("prompt", "")
        result_json = raw_example.get("result_json", "{}")
        result_data = json.loads(result_json) if isinstance(result_json, str) else result_json
        answer = result_data.get("answer", "")

        # Check length
        if len(prompt) < self.min_token_length or len(answer) < self.min_token_length:
            return False, "Example too short", raw_example

        # Deduplication
        ex_hash = self.compute_hash(prompt, answer)
        if ex_hash in self._seen_hashes:
            return False, "Duplicate example", raw_example
        self._seen_hashes.add(ex_hash)

        # Scrub PII
        clean_prompt = self.scrub_pii(prompt)
        clean_answer = self.scrub_pii(answer)

        clean_example = dict(raw_example)
        clean_example["prompt"] = clean_prompt
        clean_example["clean_answer"] = clean_answer

        return True, "Passed filter", clean_example

    def filter_batch(self, batch: List[Dict]) -> List[Dict]:
        approved = []
        for ex in batch:
            passed, reason, clean_ex = self.filter_example(ex)
            if passed:
                approved.append(clean_ex)
            else:
                logger.debug("Filtered out example: %s", reason)
        return approved
