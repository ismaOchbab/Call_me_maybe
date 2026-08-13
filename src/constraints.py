from typing import Sequence, List
from src.vocab import Vocabulary


def is_valid_prefix(value: str, choices: Sequence[str]) -> bool:
    """
    Checks whether value is a prefix of at least one allowed choice
    """
    return any(choice.startswith(value) for choice in choices)


def is_complete_choice(value: str, choices: Sequence[str]) -> bool:
    """
    Returns whether value exactly matches an allowed choice
    """
    return value in choices


def get_valid_choice_tokens_ids(
        vocab: Vocabulary,
        current_value: str,
        choices: Sequence[str]
) -> List[int]:
    """
    Return token IDs that keep current_value valid
    """
    valid_ids: List[int] = []

    for token_id, _ in vocab.items():
        token_text = vocab.token_text(token_id)

        if not token_text:
            continue

        candidate = current_value + token_text

        if is_valid_prefix(candidate, choices):
            valid_ids.append(token_id)

    return valid_ids