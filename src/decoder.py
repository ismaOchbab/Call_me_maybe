"""
Constrained decoding pipeline for function calling.

The main entry point is call_function(), which takes a user prompt,
a list of available functions, and the LLM model with its vocabulary.
It returns the chosen function name and a dictionary of arguments.

The template is filled by forcing literal tokens (braces, colons, quotes,
commas) and letting the model generate the variable parts under constraints.
"""

from typing import List, Tuple, Sequence, Any, Dict

from src.grammar import EnumConstraint, NumberConstraint, StringConstraint
from src.vocab import Vocabulary
from src.models import FunctionSchema

from src import monitor
from llm_sdk import Small_LLM_Model

# Maximum number of steps for each type of generation
MAX_ENUM_STEPS = 30
MAX_NUMBER_STEPS = 20
MAX_STRING_STEPS = 30


def pick_best(model: Small_LLM_Model,
              input_ids: List[int],
              valid_ids: List[int]) -> int:
    """
    Select the token with the highest logit from the model among the
    allowed (valid) token IDs.

    This is a greedy approach: we call the model to get logits for the
    next token given the current input_ids, and then we scan only the
    tokens in valid_ids to find the one with the highest score.

    args:
        model: The LLM model instance (must have get_logits_from_input_ids).
        input_ids: List of token IDs representing the full input so far
                   (prompt + generated tokens).
        valid_ids: List of token IDs that are allowed at this step.

    return:
        The token ID (among valid_ids) with the highest logit score.

    Raises:
        RuntimeError: If valid_ids is empty (no token available).
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


def append_literal(model: Small_LLM_Model,
                   input_ids: List[int],
                   text: str) -> List[int]:
    """
    Append a fixed (literal) string to the input_ids without calling the
    model for a decision. This is used to inject the static JSON structure

    The text is tokenized using the model's tokenizer, and the resulting
    token IDs are appended to the input list

    Args:
        model: The LLM model instance (must have encode())
        input_ids: Current token IDs
        text: The literal string to append (e.g., '{"name": "')

    Returns:
        New list of token IDs with the text appended
    """
    if not text:
        return input_ids
    new_ids: List[int] = to_id_list(model.encode(text))
    return input_ids + new_ids


def valid_enum_token_ids(constraint: EnumConstraint,
                         vocab: Vocabulary) -> List[int]:
    """
    Return all token IDs whose string representation, when appended to
    the current consumed text, keeps at least one enum candidate valid

    This is used for EnumConstraint generation (function names, booleans).

    Args:
        constraint: The EnumConstraint tracking progress.
        vocab: Vocabulary mapping token IDs to their string representation

    Returns:
        List of token IDs that are allowed under the constraint.
    """
    valid_ids: List[int] = []
    for token_id, _raw_text in vocab.items():
        text = vocab.token_text(token_id)
        if not text:
            continue
        if constraint.extends(text):
            valid_ids.append(token_id)
    return valid_ids


def generate_enum(model: Small_LLM_Model,
                  vocab: Vocabulary,
                  input_ids: List[int],
                  candidates: List[str],
                  max_steps: int = MAX_ENUM_STEPS) -> Tuple[str, List[int]]:
    """
    Generate text token by token until it exactly matches one of
    candidates

    Args:
        model: The LLM model.
        vocab: Vocabulary for token-to-text conversion.
        input_ids: Current token IDs (the context so far).
        candidates: List of allowed strings (e.g., function names).
        max_steps: Safety cap on the number of tokens.

    Returns:
        A tuple (resolved_string, updated_input_ids) where resolved_string
        is the matched candidate.
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

        monitor.step(
            len(valid_ids), len(vocab), text, constraint.consumed
        )

    return constraint.resolved_value(), ids


def valid_number_token_ids(
        constraint: NumberConstraint,
        vocab: Vocabulary,
        stop_char: str,
) -> List[int]:
    """
    Return every token id that is either:
    (a) a valid continuation of the number so far, or
    (b) the start of the fixed text that follows the number
        (`stop_char`), which is only offered once at least one
        digit has already been generated -- this is what lets the
        model "choose" to stop.

    Args:
        constraint: The NumberConstraint tracking progress
        vocab: Vocabulary for token-to-text conversion
        stop_char: The first character of the delimiter that follows the
                   number (either ',' or '}').

    Returns:
        List of allowed token IDs.
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
        model: Small_LLM_Model,
        vocab: Vocabulary,
        input_ids: List[int],
        stop_char: str,
        max_steps: int = MAX_NUMBER_STEPS
) -> Tuple[float, List[int]]:
    """
    Generate a valid JSON number token by token.

    The generation stops when the model picks a token that is not a
    valid number continuation (which should be the delimiter token).

    The delimiter is not appended, it is left for the caller to force.

    Args:
        model: The LLM model.
        vocab: Vocabulary for token-to-text conversion
        input_ids: Current context.
        stop_char: The delimiter character that should follow the number
                   (either ',' or '}').
        max_steps: Safety cap.

    Returns:
        A tuple (float_value, updated_input_ids). The float is parsed
        from the generated string
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

            monitor.step(
                len(valid_ids), len(vocab), text, constraint.buffer)
        else:
            break
    return constraint.resolved_value(), ids


def valid_string_token_ids(
        constraint: StringConstraint,
        vocab: Vocabulary
) -> List[int]:
    """
    Return token IDs that are either valid string content (no double quote)
    or the closing double quote (which the model may choose at any time).

    The closing quote is always allowed, so the model can decide to stop
    at any point.

    Args:
        constraint: The StringConstraint tracking progress.
        vocab: Vocabulary for token-to-text conversion

    Returns:
        List of allowed token IDs.
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
        model: Small_LLM_Model,
        vocab: Vocabulary,
        input_ids: List[int],
        max_steps: int = MAX_STRING_STEPS
) -> Tuple[str, List[int]]:
    """
    Generate the content of a JSON string, without the surrounding quotes.

    The generation stops when the model picks the closing quote token.

    The closing quote is not appended to the content, it is left for the
    caller to force after this function returns.

    Args:
        model: The LLM model.
        vocab: Vocabulary for token-to-text conversion
        input_ids: Current context
        max_steps: Safety cap

    Returns:
        A tuple (string_content, updated_input_ids). The content is the
        raw string (no quotes, no escaping)
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

        monitor.step(
            len(valid_ids), len(vocab), text, constraint.buffer
        )

    return constraint.resolved_value(), ids


# def build_prompt_text(
#         prompt: str,
#         functions: Sequence[FunctionSchema]
# ) -> str:
#     """Build the natural-language context shown to the model: the
#     task description, the available functions, and the user's request.
#     """
#     lines: List[str] = []

#     lines = [
#         "You are a function calling assistant.",
#         "Given the user's request,",
#         "choose the best function and provide the arguments.",
#         "Available functions:"
#     ]
#     for fn in functions:
#         param_names = ", ".join(fn.parameters.keys())
#         lines.append(f"- {fn.name}({param_names}): {fn.description}")
#     lines.append(f'User request: "{prompt}"')
#     return "\n".join(lines)

def build_prompt_text(
        prompt: str,
        functions: Sequence[FunctionSchema]
) -> str:
    """
    Build the natural-language context shown to the model.
    We explicitly instruct the model to output the function name and arguments
    in a clean, predictable format.

    The prompt includes:
        - A role description (function-calling assistant).
        - A list of available functions with their signatures and descriptions.
        - The user request prompt
        - Clear instructions for what to output: the function name and
          arguments, without extra prose.

    The last line is "Function name:" to prime the model to start generating
    the name immediately.

    Args:
        prompt: The user's natural-language request.
        functions: The available function schemas.

    Returns:
        The complete prompt string.
    """
    lines: List[str] = [
        "You are an AI assistant that extracts ",
        "function calls from user requests."
        "You have access to the following functions:"
    ]

    for fn in functions:
        # Build a detailed signature string
        params_str = ", ".join(
            [f"{k}: {v['type']}" for k, v in fn.parameters.items()])
        lines.append(f"  - {fn.name}({params_str}) -> {fn.returns['type']}")
        lines.append(f"    Description: {fn.description}")

    lines.append("")
    lines.append(f"User request: \"{prompt}\"")
    lines.append("")
    lines.append("Instructions:")
    lines.append("1. Choose the best function for the request.")
    lines.append("2. Provide the arguments exactly as requested.")
    lines.append(
        "Return the shortest exact value that satisfies the request."
        )
    lines.append(
        "Do not repeat words, patterns, alternatives, or replacement text."
        )
    lines.append("")
    lines.append("Function name:")

    return "\n".join(lines)


def call_function(
        model: Small_LLM_Model,
        vocab: Vocabulary,
        prompt: str,
        functions: Sequence[FunctionSchema]
) -> Tuple[str, Dict[str, Any]]:
    """
    Run the full constrained decoding pipeline for one user prompt.

    This function builds the prompt, forces the JSON template, and
    generates the function name and each argument value under the
    appropriate constraints. The generated JSON is guaranteed to be
    syntactically valid and schema-compliant.

    Args:
        model: The LLM model
        vocab: Vocabulary for token-to-text mapping.
        prompt: The user's natural-language request.
        functions: List of available function schemas.

    Returns:
        A tuple (chosen_function_name, parameters_dict) where
        parameters_dict maps argument names to their Python-typed values.

    Raises:
        RuntimeError: If any part of generation fails (e.g., no valid tokens).
        ValueError: If the generated number cannot be parsed or the
                    generated enum does not resolve.
    """

    # build the initial context and tokenize it
    context_text: str = build_prompt_text(prompt, functions)
    input_ids: List[int] = to_id_list(model.encode(context_text))

    # force the fixed opening of the JSON object; then let the
    # model choose the func name from known list
    input_ids = append_literal(model, input_ids, '{"name": "')
    names: List[str] = [fn.name for fn in functions]
    monitor.start(f"prompt -> {prompt}")
    chosen_name, input_ids = generate_enum(model, vocab, input_ids, names)
    monitor.done(f"Function choice -> {chosen_name}")

    # look up the schema for chosen function
    function_def: FunctionSchema = next(
        fn for fn in functions if fn.name == chosen_name)

    input_ids = append_literal(model, input_ids, '", "parameters": {')
    parameters: Dict[str, Any] = {}
    param_items = list(function_def.parameters.items())
    total_params = len(param_items)

    for index, (param_name, param_schema) in enumerate(param_items):
        is_last: bool = (index == total_params - 1)
        param_type: str = param_schema["type"]

        input_ids = append_literal(model, input_ids, f'"{param_name}": ')
        value: Any = ""

        monitor.start(f"{param_name} ({param_type})")

        if param_type in ("number", "integer"):
            stop_char = "}" if is_last else ","
            value, input_ids = generate_number(
                model, vocab, input_ids, stop_char
            )
            if param_type == "integer":
                value = int(value)
            input_ids = append_literal(model, input_ids, stop_char)
        elif param_type == "string":
            input_ids = append_literal(model, input_ids, '"')
            value, input_ids = generate_string(model, vocab, input_ids)
            input_ids = append_literal(model, input_ids, '"')
            if not is_last:
                input_ids = append_literal(model, input_ids, ", ")
            else:
                input_ids = append_literal(model, input_ids, "}")
        elif param_type == "boolean":
            value_str, input_ids = generate_enum(
                model, vocab, input_ids, ["true", "false"]
            )
            value = value_str == "true"
            if not is_last:
                input_ids = append_literal(model, input_ids, ", ")
            else:
                input_ids = append_literal(model, input_ids, "}")
        else:
            raise ValueError(
                f"Unsupported parameter type: {param_type!r}"
            )

        parameters[param_name] = value

        monitor.done(value)

    input_ids = append_literal(model, input_ids, "}")

    return chosen_name, parameters


def to_id_list(encoded: Any) -> List[int]:
    """
    Normalize the output of model.encode() into a flat list of Python ints.

    The encode method from the SDK may return a PyTorch tensor of shape
    (1, seq_len) or a list of lists. This function handles those cases
    and returns a simple List[int].

    Args:
        encoded: The output from model.encode().

    Returns:
        A flat list of token IDs.
    """
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(i) for i in encoded]
