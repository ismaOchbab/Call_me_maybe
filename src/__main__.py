"""
CLI entry point

usage:

    uv run python -m src [--functions_definition <path>] [--input <path>]
                         [--output <path>]
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List

from src.decoder import call_function
from src.io_utils import InputFileError, load_functions, load_test_prompts
from src.models import FunctionSchema, Prompt
from src.vocab import Vocabulary

from llm_sdk import Small_LLM_Model


DEFAULT_FUNCTIONS_PATH = Path("data/input/functions_definition.json")
DEFAULT_INPUT_PATH = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT_PATH = Path("data/output/function_calling_results.json")


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse cli arguments"""
    parser = argparse.ArgumentParser(prog="python3 -m src")

    parser.add_argument(
        "--functions_definition", type=Path, default=DEFAULT_FUNCTIONS_PATH
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_PATH
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH
    )

    return parser.parse_args(argv)


def load_model() -> Small_LLM_Model:
    """
    Instantiate llm_sdk wrapper with error management
    """
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError as e:
        raise InputFileError(
            "Could not import 'llm_sdk'. Make sure the package is available "
            "next to 'src', and 'uv sync' is executed"
        ) from e
    try:
        return Small_LLM_Model()
    except Exception as e:
        raise InputFileError(
            f"Could not initialize the model: {e}"
        ) from e


def run(argv: List[str]) -> int:
    """
    Run the full pipeline
    Returns : 0 = success
    """
    args = parse_args(argv)

    # load and validate input files
    try:
        functions: List[FunctionSchema] = load_functions(
            args.functions_definition
        )
        prompts: List[Prompt] = load_test_prompts(args.input)
    except InputFileError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # load the model and its vocab
    try:
        model = load_model()
        vocab = Vocabulary(model.get_path_to_vocab_file())
    except (InputFileError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # process every prompt. One bad prompt should not stop the pipeline
    # log warning and continue
    results = []
    for test_prompt in prompts:
        try:
            name, parameters = call_function(
                model, vocab, test_prompt.prompt, functions
            )
        except Exception as e:
            print(
                f"Warning: failed on '{test_prompt.prompt}': {e}",
                file=sys.stderr
            )
            continue
        results.append({
            "prompt": test_prompt.prompt,
            "name": name,
            "parameters": parameters
        })

        print(
            "Result:\n"
            f"   prompt: {test_prompt.prompt}\n"
            f"   name: {name}\n"
            f"   parameters: {parameters}\n"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(results)} results to {args.output}")
    return 0


def main() -> None:
    """CLI entry point"""
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
