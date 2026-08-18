import json
from pathlib import Path
from typing import List
from pydantic import ValidationError

from src.models import (
    FunctionSchema,
    FunctionSchemaError,
    Prompt,
    OutputSchema
)


class InputFileError(Exception):
    """JSON input file error"""
    pass


def _read_json_array(path: Path) -> list:
    """ reads a JSON array from disk"""
    if not path.is_file():
        raise InputFileError(
            f"File not found: {path}"
        )
    try:
        with path.open(mode='r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise InputFileError(
            f"{path} is not valid JSON: {e}"
        ) from e
    if not isinstance(data, list):
        raise InputFileError(
            f"{path} must contain a JSON array"
        )
    return data


def load_functions(path: Path) -> List[FunctionSchema]:
    """
    Loads the functions definitions file as List of Dict
    """
    raw_items = _read_json_array(path)
    functions = []
    for i, item in enumerate(raw_items):
        try:
            functions.append(FunctionSchema.model_validate(item))
        except (ValidationError, FunctionSchemaError) as e:
            raise InputFileError(
                f"{path}: entry {i} is invalid: {e}"
            ) from e
    return functions


def load_test_prompts(path: Path) -> List[Prompt]:
    """
    loads the test prompts file as a list of Prompt
    """
    raw_items = _read_json_array(path)
    prompts = []
    for i, item in enumerate(raw_items):
        try:
            prompts.append(Prompt.model_validate(item))
        except ValidationError as e:
            raise InputFileError(
                f"{path}: entry {i} is invalid: {e}"
            ) from e
    return prompts


def write_results(path: Path, results: OutputSchema) -> None:
    """
    Writes the generated function calls as a JSON array
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode='w', encoding='utf-8') as f:
            json.dump(
                [result.model_dump() for result in results],
                f,
                indent=2,
                ensure_ascii=False
            )
            f.write("\n")
    except OSError as e:
        raise InputFileError(
            f"Cannot write {path}: {e}"
        ) from e