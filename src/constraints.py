from typing import Sequence, List, Set
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


def mask_logits(
        logits: Sequence[float],
        valid_token_ids: Set[int],
) -> List[float]:
    """
    Mask invalid token logits with negative infinity
    """
    masked_logits: List[float] = []

    for token_id, logit in enumerate(logits):
        if token_id in valid_token_ids:
            masked_logits.append(logit)
        else:
            masked_logits.append(float("-inf"))

    return masked_logits


def select_best_token(masked_logits: Sequence[float]) -> int:
    """
    Return the token ID with the highest valid logit
    """
    if not masked_logits:
        raise ValueError(
            f"Cannot select a token from empty logits"
        )

    best_token_id = max(
        range(len(masked_logits)),
        key=lambda token_id: masked_logits[token_id]
    )

    if masked_logits[best_token_id] == float("-inf"):
        raise ValueError(
            "No valid token available"
        )

    return best_token_id