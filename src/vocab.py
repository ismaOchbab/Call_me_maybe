import json
from pathlib import Path
from typing import Dict, ItemsView


_SPACE_MARKERS = ("\u0120", "\u2581")


class Vocabulary:
    """
    Lookup table: token id -> normalized text
    """
    def __init__(self, vocab_path: str) -> None:
        self._id_to_token: Dict[int, str] = {}
        self._load(vocab_path)

    def _load(self, vocab_path: str) -> None:
        path = Path(vocab_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocab_path}"
            )
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            for token_str, token_id in raw.items():
                self._id_to_token[int(token_id)] = token_str
        elif isinstance(raw, list):
            for token_id, token_str in enumerate(raw):
                self._id_to_token[token_id] = str(token_str)
        else:
            raise ValueError("Unsupported vocab file format")

    def token_text(self, token_id: int) -> str:
        """
        Returns the normalized (clean) text for a token_id
        returns "" for unknown id
        """
        raw = self._id_to_token.get(token_id)
        if raw is None:
            return ""
        return self.normalize_token_text(raw)

    @staticmethod
    def normalize_token_text(raw_token: str) -> str:
        """
        Turn a raw vocab entry into the literal text it represents
        """
        for marker in _SPACE_MARKERS:
            if raw_token.startswith(marker):
                return " " + raw_token[len(marker):]
        return raw_token

    def items(self) -> ItemsView[int, str]:
        """
        (id, raw_text) pairs for the whole vocab
        used by the decoder to scan every possible next token
        """
        return self._id_to_token.items()

    def __len__(self) -> int:
        """Returns the length of vocab"""
        return len(self._id_to_token)