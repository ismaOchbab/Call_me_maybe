#!/usr/bin/env python3

import json
from pathlib import Path

from src.models import FunctionSchema
import src.io_utils

# raw = json.load(open("data/input/functions_definition.json"))

# fn = FunctionSchema.model_validate(raw[0])

# print(fn)

print(src.io_utils.load_functions(Path("data/input/functions_definition.json")))