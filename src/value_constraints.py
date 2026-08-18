import json
from typing import Any, Set

from src.vocab import Vocabulary
from src.constraints import mask_logits, select_best_token, GenerationError
from llm_sdk import Small_LLM_Model


_BOOLEAN_VALUES = ("true", "false")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def is_valid_boolean_prefix(value: str) -> bool:
    """
    return whether value can become a valid JSON boolean
    """
    return any(item.startswith(value) for item in _BOOLEAN_VALUES)


def is_complete_boolean(value: str) -> bool:
    """
    Return whether value is a complete JSON boolean
    """
    return value in _BOOLEAN_VALUES


def is_valid_number_prefix(value: str) -> bool:
    """Return whether value can become a valid JSON number."""
    if value == "":
        return True

    state = "start"

    for char in value:
        if state == "start":
            if char == "-":
                state = "sign"
            elif char == "0":
                state = "zero"
            elif char in "123456789":
                state = "integer"
            else:
                return False

        elif state == "sign":
            if char == "0":
                state = "zero"
            elif char in "123456789":
                state = "integer"
            else:
                return False

        elif state == "zero":
            if char == ".":
                state = "decimal_start"
            elif char in "eE":
                state = "exponent_start"
            else:
                return False

        elif state == "integer":
            if char.isdigit():
                continue
            if char == ".":
                state = "decimal_start"
            elif char in "eE":
                state = "exponent_start"
            else:
                return False

        elif state == "decimal_start":
            if char.isdigit():
                state = "decimal"
            else:
                return False

        elif state == "decimal":
            if char.isdigit():
                continue
            if char in "eE":
                state = "exponent_start"
            else:
                return False

        elif state == "exponent_start":
            if char in "+-":
                state = "exponent_sign"
            elif char.isdigit():
                state = "exponent"
            else:
                return False

        elif state == "exponent_sign":
            if char.isdigit():
                state = "exponent"
            else:
                return False

        elif state == "exponent":
            if not char.isdigit():
                return False

    return True


def is_complete_number(value: str) -> bool:
    """
    Return whether value is a valide JSON number
    """
    if not is_valid_number_prefix(value):
        return False

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False

    return isinstance(parsed, (int, float)) and not isinstance(parsed, bool)


def is_valid_string_prefix(value: str) -> bool:
    """
    Return true if value can become a valid JSON string
    """
    if value == "":
        return True

    if value[0] != '"':
        return False

    escaped = False
    unicode_digits = 0
    in_unicode_escape = False

    for index, char in enumerate(value[1:], start=1):
        if in_unicode_escape:
            if char not in _HEX_DIGITS:
                return False
            unicode_digits += 1
            if unicode_digits == 4:
                in_unicode_escape = False
            continue

        if escaped:
            escaped = False
            if char == "u":
                in_unicode_escape = True
                unicode_digits = 0
            elif char not in '"\\/bfnrt':
                return False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == '"':
            return index == len(value) - 1

        if ord(char) < 0x20:
            return False

    return True


def is_complete_string(value: str) -> bool:
    """
    Return true if value is a complete valid JSON string
    """
    if not is_valid_string_prefix(value):
        return False

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False

    return isinstance(parsed, str)


def is_valid_value_prefix(value: str, parameter_type: str) -> bool:
    """
    Returns True if value is a valid prefix for its schema type
    """
    if parameter_type == "boolean":
        return is_valid_boolean_prefix(value)

    if parameter_type == "number":
        return is_valid_number_prefix(value)

    if parameter_type == "string":
        return is_valid_string_prefix(value)

    return False


def is_complete_value(value: str, parameter_type: str) -> bool:
    """
    Returns True if value is complete for its schema type
    """
    if parameter_type == "boolean":
        return is_complete_boolean(value)

    if parameter_type == "number":
        return is_complete_number(value)

    if parameter_type == "string":
        return is_complete_string(value)

    return False


def get_valid_value_token_ids(
        vocab: Vocabulary,
        current_value: str,
        parameter_type: str
) -> Set[int]:
    """
    Return vocabulary IDs that preserve value validity
    """
    valid_ids = set()

    for token_id, _ in vocab.items():
        token_text = vocab.token_text(token_id)

        if not token_text:
            continue

        candidate = current_value + token_text

        if is_valid_value_prefix(candidate, parameter_type):
            valid_ids.add(token_id)

    return valid_ids


def parse_value(value: str, parameter_type: str) -> Any:
    """
    Convert a complete JSON value to its Python representation
    """
    if not is_complete_value(value, parameter_type):
        raise ValueError(
            f"Invalid '{parameter_type}' for value: '{value}'"
        )

    return json.loads(value)


ParameterValue = str | float | int | bool
def generate_parameter_value(
        model: Small_LLM_Model,
        vocab: Vocabulary,
        prompt: str,
        parameter_type: str,
        delimiter: str,
        max_tokens: int = 100
) -> ParameterValue:
    """
    Generate one schema-constrained parameter value
    """
    input_ids: list[int] = model.encode(prompt).tolist()[0]
    generated = ""

    for _ in range(max_tokens):
        valid_ids = set()

        for token_id, _ in vocab.items():
            token_text = vocab.token_text(token_id)

            if not token_text:
                continue

            if is_valid_value_prefix(
                generated + token_text,
                parameter_type
            ):
                valid_ids.add(token_id)
            elif (
                token_text == delimiter
                and is_complete_value(generated, parameter_type)
            ):
                valid_ids.add(token_id)

        if not valid_ids:
            raise GenerationError(
                f"No valid token for {parameter_type}: "
                f"'{generated}'"
            )

        logits = model.get_logits_from_input_ids(input_ids)
        masked_logits = mask_logits(logits, valid_ids)
        next_token_id = select_best_token(masked_logits)
        next_token_text = vocab.token_text(next_token_id)

        input_ids.append(next_token_id)

        if (
            next_token_text == delimiter
            and is_complete_value(generated, parameter_type)
        ):
            return parse_value(generated, parameter_type)

        generated += vocab.token_text(next_token_id)

        if is_complete_value(generated, parameter_type):
            return parse_value(generated, parameter_type)

    raise GenerationError(
        f"Parameter generation exceeded {max_tokens} tokens"
    )