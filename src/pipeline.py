"""
Orchestration of the two constrained-decoding stages

stage 1: picks the function name out of the allowed choices
stage2: fills every parameter of that function, one at a time,
    each one constrained on the JSON type declared in the schema
"""

import json
from typing import Dict, List, Sequence

from llm_sdk import Small_LLM_Model

from src.constraints import (
    GenerationError,
    build_model_prompt,
    generate_choice
)
from src.models import FunctionSchema, OutputSchema, Prompt
from src.value_constraints import (
    ParameterValue,
    generate_parameter_value
)
from src.vocab import Vocabulary


def build_json_prefix(
        function_name: str,
        filled_parameters: Dict[str, ParameterValue],
        next_parameter: str
)-> str:
    """
    Render the JSON produced so far, left open on next_parameter
    For example:
        '{"name": "fn_add_numbers", "parameters": {"a": 2, "b": '
        The model sees real, syntactically coherent JSON and only has to
        continue it, which is what the constraint then enforces
    """
    rendered = ", ".join(
        f"{json.dumps(key)}: {json.dumps(value)}"
        for key, value in filled_parameters.items()
    )
    if rendered:
        rendered += ", "

    return (
        f"{{{json.dumps('name')}: {json.dumps(function_name)}, "
        f"{json.dumps('parameters')}: "
        f"{{{rendered}{json.dumps(next_parameter)}:"
    )


def build_parameter_prompt(
        user_prompt: str,
        function: FunctionSchema,
        filled_parameters: Dict[str, ParameterValue],
        next_parameter: str
) -> str:
    """
    Build the prompt used to generate one parameter value
    """
    parameter_type = function.parameters[next_parameter]["type"]

    return (
        "Extract the arguments of the function call as JSON.\n"
        f"Function: {function.name}: {function.description}\n"
        f"Parameters: {function.parameters}\n"
        f"User request: {user_prompt}\n"
        f"The value of '{next_parameter}' is a {parameter_type}"
        + build_json_prefix(
            function.name,
            filled_parameters,
            next_parameter
        )
    )


def generate_parameters(
        model: Small_LLM_Model,
        vocab: Vocabulary,
        function: FunctionSchema,
        user_prompt: str
) -> Dict[str, ParameterValue]:
    """
    Generate every parameter of funcion, in schema order
    """
    parameter_names = list(function.parameters.keys())
    filled: Dict[str, ParameterValue] = {}

    for index, name in enumerate(parameter_names):
        is_last = index == len(parameter_names) - 1

        filled[name] = generate_parameter_value(
            model=model,
            vocab=vocab,
            prompt=build_parameter_prompt(
                user_prompt=user_prompt,
                function=function,
                filled_parameters=filled,
                next_parameter=name
            ),
            parameter_type=function.parameters[name]["type"],
            delimiter="}" if is_last else ","
        )

    return filled


def generate_function_call(
        model: Small_LLM_Model,
        vocab: Vocabulary,
        functions: Sequence[FunctionSchema],
        user_prompt: str
) -> OutputSchema:
    """
    Turn one natural-language prompt into one function call
    """
    function_name = generate_choice(
        model=model,
        vocab=vocab,
        prompt=build_model_prompt(
            user_prompt=user_prompt,
            functions=list(functions)
        ),
        choices=[function.name for function in functions]
    )

    by_name = {function.name: function for function in functions}
    function = by_name[function_name]

    return OutputSchema(
        prompt=user_prompt,
        name=function_name,
        parameters=generate_parameters(
            model=model,
            vocab=vocab,
            function=function,
            user_prompt=user_prompt
        )
    )


def run_pipeline(
        model: Small_LLM_Model,
        vocab: Vocabulary,
        functions: Sequence[FunctionSchema],
        prompts: Sequence[Prompt]
) -> List[OutputSchema]:
    """
    Process every prompt, skipping the ones that cannot generate
    """
    results = []

    for item in prompts:
        try:
            results.append(
                generate_function_call(
                    model=model,
                    vocab=vocab,
                    functions=functions,
                    user_prompt=item.prompt
                )
            )
        except (GenerationError, ValueError) as error:
            print(
                f"Skipping '{item.prompt}': {error}"
            )

    return results