"""
I/O utilities for loading and saving JSON data.

This module provides functions to read function definitions and test prompts
from JSON files, validate them using Pydantic models
"""


import json

from pathlib import Path
from typing import List, Any
from pydantic import ValidationError

from src.models import (
    FunctionSchema,
    FunctionSchemaError,
    Prompt
)


class InputFileError(Exception):
    """Raised when an input file is missing, malformed, or invalid."""
    pass


def _read_json_array(path: Path) -> List[Any]:
    """
    Read and parse a JSON file that must contain a top-level array.

    This is a helper used by both load_functions and load_test_prompts.
    It handles file existence checks, JSON decoding, and type validation.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON array (list of objects).

    Raises:
        InputFileError: If the file is missing, not valid JSON, or not an array
    """
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
    Load and validate the function definitions file

    Args:
        path: Path to the functions definition file

    Returns:
        List of validated FunctionSchema objects

    Raises:
        InputFileError: If any entry fails validation.
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
    Load and validate the test prompts file.

    Args:
        path: Path to the prompts file

    Returns:
        List of validated Prompt objects

    Raises:
        InputFileError: If any entry fails validation
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
