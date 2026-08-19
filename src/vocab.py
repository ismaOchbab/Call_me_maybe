"""
Vocabulary mapping for token IDs to human-readable text.

This module provides a Vocabulary class that loads a tokenizer's vocabulary
file and allows lookups from token ID to the actual string representation
of that token, with special handling for space markers commonly used by
subword tokenizers like SentencePiece or BPE.
"""


import json
from pathlib import Path
from typing import Dict, ItemsView


_SPACE_MARKERS = ("\u0120", "\u2581")


class Vocabulary:
    """
    Lookup table: token id -> normalized text
    """
    def __init__(self, vocab_path: str) -> None:
        """
        Initialize the vocabulary by loading the file at the given path.

        Args:
            vocab_path: Path to the vocabulary file (JSON format).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is neither dict nor list.
        """
        self._id_to_token: Dict[int, str] = {}
        self._load(vocab_path)

    def _load(self, vocab_path: str) -> None:
        """
        Load the vocabulary file and fill the internal mapping.

        The file must be a JSON file with either:
            - A dict mapping token strings to integer IDs, or
            - A list of token strings where the index is the ID.

        Args:
            vocab_path: Path to the JSON file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON root is neither dict nor list.
        """
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
        Returns the normalized text representation for a given token ID.

        The normalization removes space markers (e.g., "Ġ" or "▁") and
        replaces them with a regular space character.

        Args:
            token_id: The integer ID of the token.

        Returns:
            The normalized string representation, or "" if the ID is unknown
        """
        raw = self._id_to_token.get(token_id)
        if raw is None:
            return ""
        return self.normalize_token_text(raw)

    @staticmethod
    def normalize_token_text(raw_token: str) -> str:
        """
        Convert a raw vocabulary entry into the literal text it represents.

        Args:
            raw_token: The raw string from the vocabulary file.

        Returns:
            The normalized string with space markers replaced by spaces
        """
        for marker in _SPACE_MARKERS:
            if raw_token.startswith(marker):
                return " " + raw_token[len(marker):]
        return raw_token

    def items(self) -> ItemsView[int, str]:
        """
        Return all (token_id, raw_text) pairs in the vocabulary.
        The raw_text is NOT normalized

        Returns:
            A view of (token_id, raw_token_string) pairs
        """
        return self._id_to_token.items()

    def __len__(self) -> int:
        """
        Returns the length of vocab
        """
        return len(self._id_to_token)
