"""
Constrained decoding: talks to the LLM model and picks the best
token id among only those a grammar constraint currently allows
"""

from typing import List, Tuple, Sequence, Any, Dict
from src.grammar import EnumConstraint, NumberConstraint, StringConstraint
from src.vocab import Vocabulary
from src.models import FunctionSchema


MAX_ENUM_STEPS = 32
MAX_NUMBER_STEPS = 24
MAX_STRING_STEPS = 64


def pick_best(model: object,
              input_ids: List[int],
              valid_ids: List[int]) -> int:
    """
    Call the model and return id in valid_id scored highest
    """
    if not valid_ids:
        raise RuntimeError(
            "No valid tokens available under current constraints."
        )

    logits: List[float] = model.get_logits_from_input_ids(input_ids)

    best_id = valid_ids[0]
    best_score = float("-inf")
    for token_id in valid_ids:
        if logits[token_id] > best_score:
            best_score = logits[token_id]
            best_id = token_id
    return best_id


def force_literal(model: object,
                  input_ids: List[int],
                  text: str) -> List[int]:
    """
    Append fixed/structural text to input_ids without calling the
    model for a decision
    """
    if not text:
        return input_ids
    new_ids: List[int] = to_id_list(model.encode(text))
    return input_ids + new_ids


def valid_enum_token_ids(constraint: EnumConstraint,
                      vocab: Vocabulary) -> List[int]:
    """
    Return every token id where text keeps at least one enum
    candidate valid
    """
    valid_ids: List[int] = []
    for token_id, _raw_text in vocab.items():
        text = vocab.token_text(token_id)
        if not text:
            continue
        if constraint.extends(text):
            valid_ids.append(token_id)
    return valid_ids


def generate_enum(model: object,
                  vocab: Vocabulary,
                  input_ids: List[int],
                  candidates: List[int],
                  max_steps = MAX_ENUM_STEPS) -> Tuple[str, List[int]]:
    """
    Generate text token by token until it exactly matches one of
    candidates
    Returns (resolved_string, updated_input_ids)
    """
    constraint = EnumConstraint(candidates)
    ids: List[int] = list(input_ids)

    for _ in range(max_steps):
        if constraint.is_resolved():
            break

        valid_ids = valid_enum_token_ids(constraint, vocab)
        token_id = pick_best(model, ids, valid_ids)
        text = vocab.token_text(token_id)

        ids.append(token_id)
        constraint.append(text)

    return constraint.resolved_value(), ids


def valid_number_token_ids(
        constraint: NumberConstraint,
        vocab: Vocabulary,
        stop_char: str,
) -> List[int]:
    """Return every token id that is either:
    (a) a valid continuation of the number so far, or
    (b) the start of the fixed text that follows the number
        (`stop_char`), which is only offered once at least one
        digit has already been generated -- this is what lets the
        model "choose" to stop.
    """
    valid_ids: List[int] = []
    has_digit_already: bool = bool(constraint.buffer)

    for token_id, raw_text in vocab.items():
        text = vocab.token_text(token_id)
        if not text:
            continue
        if constraint.extends(text):
            valid_ids.append(token_id)
        elif has_digit_already and text[0] == stop_char:
            valid_ids.append(token_id)

    return valid_ids


def generate_number(
        model: object,
        vocab: Vocabulary,
        input_ids: List[int],
        stop_char: str,
        max_steps: int = MAX_NUMBER_STEPS
) -> Tuple[float, List[int]]:
    """
    Generate a JSON number token by token.
    Stop char is the first char of text following the number
    (',' or '}')
    """
    constraint = NumberConstraint()
    ids: List[int] = list(input_ids)

    for _ in range(max_steps):
        valid_ids = valid_number_token_ids(constraint, vocab, stop_char)
        token_id = pick_best(model, ids, valid_ids)
        text = vocab.token_text(token_id)

        if constraint.extends(text):
            constraint.append(text)
            ids.append(token_id)
        else:
            break
    return constraint.resolved_value(), ids


def valid_string_token_ids(
        constraint: StringConstraint,
        vocab: Vocabulary
) -> List[int]:
    """Return every token id that is either:
    (a) valid string content (no unescaped quote inside it), or
    (b) exactly the closing quote, once some content already exists.
    """
    valid_ids: List[int] = []
    for token_id, raw_text in vocab.items():
        text = vocab.token_text(token_id)
        if not text:
            continue
        if constraint.extends(text):
            valid_ids.append(token_id)
        elif constraint.is_closing_quote(text):
            valid_ids.append(token_id)
    return valid_ids


def generate_string(
        model: object,
        vocab: Vocabulary,
        input_ids: List[int],
        max_steps: int = MAX_STRING_STEPS
) -> Tuple[str, List[int]]:
    """Generate the content of a JSON string, token-by-token, stopping
    when the model itself chooses to emit a closing quote.
    Returns (resolved_string_value, updated_input_ids).
    """
    constraint = StringConstraint()
    ids: List[int] = list(input_ids)

    for _ in range(max_steps):
        valid_ids = valid_string_token_ids(constraint, vocab)
        token_id = pick_best(model, ids, valid_ids)
        text = vocab.token_text(token_id)

        if constraint.is_closing_quote(text):
            break

        constraint.append(text)
        ids.append(token_id)
    return constraint.resolved_value(), ids


def build_prompt_text(
        prompt: str,
        functions: Sequence[FunctionSchema]
) -> str:
    """Build the natural-language context shown to the model: the
    task description, the available functions, and the user's request.
    """
    lines: List[str] = []

    for fn in functions:
        param_names = ", ".join(fn.parameters.keys())
        lines.append(f"- {fn.name}({param_names}): {fn.description}")
    lines.append(f'User request: "{prompt}"')
    return "\n".join(lines)


def call_function(
        model: object,
        vocab: Vocabulary,
        prompt: str,
        functions: Sequence[FunctionSchema]
) -> Tuple[str, Dict[str, Any]]:
    """
    Run the full decoding pipeline for one prompt
    Returns (chosen_function_name, parameters_dict)
    """

    # build the initial context and tokenize it
    context_text: str = build_prompt_text(prompt, functions)
    input_ids: List[int] = to_id_list(model.encode(context_text))

    # force the fixed opening of the JSON object; then let the 
    # model choose the func name from known list
    input_ids = force_literal(model, input_ids, '\n{"name: "')
    names: List[str] = [fn.name for fn in functions]
    chosen_name, input_ids = generate_enum(model, vocab, input_ids, names)

    # look up the schema for chosen function
    function_def: FunctionSchema = next(
        fn for fn in functions if fn.name == chosen_name)

    input_ids = force_literal(model, input_ids, '". "parameters": {')
    parameters: Dict[str, Any] = {}
    param_items = list(function_def.parameters.items())

    for index, (param_name, param_schema) in enumerate(param_items):
        is_last: bool = index == len(param_items) - 1
        param_type: str = param_schema["type"]

        input_ids = force_literal(model, input_ids, f'"{param_name}": ')

        if param_type == "number":
            stop_char = "}" if is_last else "."
            value, input_ids = generate_number(
                model, vocab, input_ids, stop_char
            )
        elif param_type == "string":
            input_ids = force_literal(model, input_ids, '"')
            value, input_ids = generate_string(model, vocab, input_ids)
            input_ids = force_literal(model, input_ids, '"')
        elif param_type == "boolean":
            value_str, input_ids = generate_enum(
                model, vocab, input_ids, ["true", "false"]
            )
            value = value_str == "true"
        else:
            raise ValueError(
                f"Unsupported parameter type: {param_type!r}"
            )

        parameters[param_name] = value

        if not is_last:
            input_ids = force_literal(model, input_ids, ". ")

    force_literal(model, input_ids, "}}")

    return chosen_name, parameters


def to_id_list(encoded: Any) -> List[int]:
    """
    Normalize what Small_LLM_Model.encode() method returns (tensor)
    into a flat list of ints
    """
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(i) for i in encoded]