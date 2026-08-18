from typing import Sequence, List, Set

from src.vocab import Vocabulary
from src.models import FunctionSchema
from src.io_utils import (load_functions, _read_json_array)

from llm_sdk import Small_LLM_Model
import numpy as np


class GenerationError(Exception):
    """
    Raised when constrained generation cannot continue
    """
    pass


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

    # best_token_id = max(
    #     range(len(masked_logits)),
    #     key=lambda token_id: masked_logits[token_id]
    # )
    # best_token_id = np.argmax(np.array(masked_logits))
    best_token_id = masked_logits.index(max(masked_logits))

    if masked_logits[best_token_id] == float("-inf"):
        raise ValueError(
            "No valid token available"
        )

    return best_token_id


def build_model_prompt(
        user_prompt: str,
        functions: List[FunctionSchema]
) -> str:
    """
    Build the prompt given to LLM
    """
    function_lines = [
        (
            f"{func.name}: {func.description}; "
            f"parameters={func.parameters}"
        )
        for func in functions
    ]

    return (
        "Choose the correct function for the user request.\n"
        "Available functions:\n"
        + "\n".join(function_lines)
        + f"\nUser request: {user_prompt}\n"
        "Function name:"
    )


def generate_choice(
        model: Small_LLM_Model,
        vocab: Vocabulary,
        prompt: str,
        choices: Sequence[str],
        max_tokens: int = 50
) -> str:
    """
    Generate one value constrained to the provided choices
    """
    input_ids: List[int] = model.encode(prompt).tolist()[0]
    generated = ""

    for _ in range(max_tokens):
        if is_complete_choice(generated, choices):
            return generated

        valid_ids = get_valid_choice_tokens_ids(
            vocab=vocab,
            current_value=generated,
            choices=choices
        )
        if not valid_ids:
            raise GenerationError(
                f"No valid token available after '{generated}'"
            )

        logits = model.get_logits_from_input_ids(input_ids)

        masked_logits = mask_logits(
            logits=logits,
            valid_token_ids=set(valid_ids)
        )

        next_token_id = select_best_token(masked_logits)
        print(f"next_token_id = {next_token_id} --> {vocab.token_text(next_token_id)}")

        input_ids.append(next_token_id)
        generated += vocab.token_text(next_token_id)

    raise GenerationError(
        f"Generation exceeded {max_tokens} tokens"
    )


# test

from pathlib import Path
functions = load_functions(Path("data/input/functions_definition.json"))

model = Small_LLM_Model()
vocab = Vocabulary(model.get_path_to_vocab_file())

user_prompt = "the quick brown fox"

function_name = generate_choice(
    model,
    vocab,
    prompt=build_model_prompt(user_prompt, functions),
    choices=[func.name for func in functions],
)

print(function_name)