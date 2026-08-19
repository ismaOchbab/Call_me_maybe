*This project has been created as part of the 42 curriculum by ichbab.*

# Call_me_maybe

## Description

A function calling tool that turns natural language into structured function
calls.

Given `"What is the sum of 2 and 3?"`, it does not answer `5` — it
returns the function to call and its typed arguments:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The output is always valid, schema-compliant JSON. That guarantee does not
come from prompting the model nicely, nor from repairing its text
afterwards. It comes from **constrained decoding**: at every generation
step, tokens that would break the JSON structure or violate the parameter's
declared type are excluded before the next token is chosen. An invalid
result is never produced in the first place.

The model is `Qwen/Qwen3-0.6B`, accessed only through the public methods of
the provided `llm_sdk`.

### Project layout

```
src/
  __main__.py   CLI entry point
  decoder.py    constrained decoding + call_function() orchestration
  grammar.py    the three value constraints
  vocab.py      token id -> literal text lookup
  models.py     pydantic schemas for validation
  io_utils.py   JSON loading and validation
  monitor.py    step-by-step generation trace
```

## Instructions

```bash
make install       # uv sync
make run           # run on the default input files
make debug         # run under pdb
make lint          # flake8 + mypy
make lint-strict   # flake8 + mypy --strict
make clean         # remove caches and data/output
```

Direct invocation:

```bash
uv run python -m src [--functions_definition <path>] \
                     [--input <path>] \
                     [--output <path>]
```

Defaults: reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, writes
`data/output/function_calling_results.json`. The output directory is created
if it does not exist.

## Example usage

```bash
$ uv run python -m src
prompt -> What is the sum of 265 and 345?
      31/151643 'fn'       -> fn
       4/151643 '_add'     -> fn_add
       1/151643 '_numbers' -> fn_add_numbers
  = Function choice -> fn_add_numbers

a (number)
     213/151643 '2'  -> 2
      12/151643 '65' -> 265
  = 265.0

b (number)
     486/151643 '3'  -> 3
     112/151643 '45' -> 345
  = 265.0
Result:
   prompt: What is the sum of 265 and 345?
   name: fn_add_numbers
   parameters: {'a': 265.0, 'b': 345.0}

...

Wrote 11 results to data/output/function_calling_results.json
```

Custom paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

## Algorithm explanation

### The core idea

A JSON function call is mostly fixed text. In
`{"name": "fn_greet", "parameters": {"name": "shrek"}}`, only `fn_greet` and
`shrek` are decisions — every brace, quote, colon and comma is structure
that follows mechanically from the schema.

The pipeline splits the output along exactly that line:

- **Structural text is forced.** `append_literal()` tokenizes the fixed
  fragment and appends the token IDs straight to the context, without ever
  consulting the model. The model cannot misplace a brace it was never
  asked to produce.
- **Values are generated under constraint.** Only the variable parts go
  through the model, each restricted to the grammar of its declared type.

### One constrained step

For each token of a value:

1. Scan the whole vocabulary. For each token, ask the active constraint
   whether appending its text keeps the value valid so far. This builds the
   list of allowed token IDs — the mask.
2. Call `get_logits_from_input_ids()` for the current context.
3. `pick_best()` selects the highest logit **among the allowed IDs only**,
   which is equivalent to setting every other logit to negative infinity.
4. Append the chosen token and repeat.

The model still expresses its preference; it simply cannot express an
invalid one.

### The three constraints (`src/grammar.py`)

Each constraint keeps a buffer of what has been generated and answers
`extends(text)` — "would appending this token keep the value valid?" —
plus `resolved_value()`, which converts the buffer to a Python value.

- **`EnumConstraint`** — the value must exactly match one of a fixed list of
  candidates. Each appended token narrows the candidate list to those still
  starting with what has been consumed. It resolves when exactly one
  candidate remains and equals the consumed text. Used for the function name
  and for booleans.
- **`NumberConstraint`** — the buffer must stay a valid JSON number prefix.
  `""`, `"-"`, `"3"`, `"3."` are valid prefixes; `".5"` and `"1.2.3"` are
  not. `resolved_value()` returns a float, and raises `ValueError` if the
  buffer is not a complete number.
- **`StringConstraint`** — the buffer is the *content* of a JSON string,
  without the surrounding quotes. A token is valid content if it contains no
  double quote, which sidesteps the need for the model to escape anything.

### Knowing when to stop

Each type ends differently:

- **Enum** — self-terminating. The loop checks `is_resolved()` at the top
  and stops as soon as the consumed text equals a single surviving
  candidate.
- **String** — the closing quote is *always* included in the mask, so the
  model itself decides when the string is finished. The quote is not
  appended to the buffer; the caller forces it afterwards.
- **Number** — **not** self-terminating. `2` is a complete JSON number, but
  so is `265`. Stopping as soon as the buffer parses would truncate `265` to
  `2`. Instead, `valid_number_token_ids()` adds the delimiter that follows
  the number (`,` or `}`) to the mask, but only once at least one digit
  exists. The model choosing that delimiter is the stop signal.

### Full pipeline (`src/decoder.py`, `call_function`)

```
build_prompt_text()               task description + function signatures
  -> append_literal('{"name": "')
  -> generate_enum(...)           constrained to the function names
  -> append_literal('", "parameters": {')
  -> for each parameter, in schema order:
       append_literal('"<param>": ')
       generate_number / generate_string / generate_enum
       append_literal(',' or '}')
  -> append_literal('}')
```

The chosen name is looked up in the loaded schemas, so the parameter list,
their order and their types all come from `functions_definition.json` — no
function or argument is ever hardcoded.

## Design decisions

**Forcing structure instead of validating it.** Letting the model emit the
entire JSON and constraining every character would require a full JSON
parser as a state machine. Forcing the skeleton reduces the problem to three
small, independently testable value grammars.

**Greedy selection, not sampling.** `pick_best()` takes the argmax over the
allowed tokens.

**The vocabulary file drives the mask.** The mask is built from
`get_path_to_vocab_file()`, which maps token IDs to their literal text.
`encode()` is used only to append forced literals.

**Schemas are validated before decoding starts.** `FunctionSchema` is a
pydantic model with `extra: "forbid"` and a `model_validator` that rejects
missing or unsupported types. A malformed function definition fails at load
time with a clear message, so the decoder never runs against a bad schema.

**`integer` is supported alongside `number`.** Both decode with
`NumberConstraint`; `integer` is cast with `int()` afterwards.

**Step caps on every generation.** `MAX_ENUM_STEPS = 30`,
`MAX_NUMBER_STEPS = 20`, `MAX_STRING_STEPS = 30` bound the work per value so
a confused model cannot generate indefinitely.

**Errors never stop the run.** Each prompt is processed in its own
`try/except` in `__main__.run`. A failure is reported on stderr and the
remaining prompts continue, so one difficult request cannot cost the whole
output file.

## Performance analysis

Measured against a vocabulary of 151,643 tokens (Qwen3-0.6B size):

The mask scan is the dominant per-step cost, since it touches every token in
the vocabulary at every generation step. Across the 11 provided prompts this
totals roughly 20 seconds of masking; the rest of the runtime is the model's
forward passes, one per generated token. The full set completes well inside
the 5-minute requirement.

**Accuracy** on the provided test set: **11/11 correct function selection**, **8/11 fully correct arguments**. All three imperfect cases
are `fn_substitute_string_with_regex`, where the `regex` parameter must be
*invented* (`[aeiouAEIOU]`) rather than copied from the request. The
function name, `source_string` and `replacement` are correct in all three.

This is the important distinction to keep in mind: constrained decoding
guarantees **structural and type validity**, never **semantic correctness**.
All 11 outputs are valid, parseable, schema-compliant JSON with every
required argument present and correctly typed. Whether the value inside is
the *right* value is the model's judgement, and that is what prompt quality
influences.

**Reliability** is total across runs. Greedy selection over a deterministic
mask means identical input always produces identical output.

## Challenges faced

**Numbers truncated to a single digit.** `265` came out as `2`. The cause
was treating "the buffer parses as a JSON number" as "the value is
finished" — true for strings and enums, false for numbers, since every
prefix of a number is also a number. Solved by making the following
delimiter the only stop signal, offered in the mask only once a digit
exists.

**Tokenizer artifacts leaking into values.** Early output contained
sequences like `}}ĊAnswer:Ċ```jsonĊ{`. Those are byte-level BPE markers,
where `Ġ` encodes a space and `Ċ` a newline, mixed with the model trying to
answer in its chat format. Two things fixed it: forcing the JSON skeleton so
the model is never in a position to open a code fence, and giving each value
a constraint tight enough that the surrounding noise is unreachable.

**Weak prompt, weak arguments.** The first prompt produced placeholder
values like `"description of source string here"` — the model was completing
a template rather than extracting from the request. Rewriting
`build_prompt_text()` to show typed signatures, return types, and explicit
instructions to return the shortest exact value fixed the simple cases
entirely.


## Known limitations

Documented deliberately rather than hidden — each is understood, bounded,
and would be the first thing to address next.

- **Escape sequences are not decoded.** `StringConstraint` stores the raw
  buffer, so a generated `\s` is written as `\\s`. This is visible in the
  regex arguments of the test output. Fixing it means validating escapes in
  `extends()` and decoding the buffer through `json.loads` in
  `resolved_value()`.
- **Step-cap exhaustion returns a partial value.** If a string reaches
  `MAX_STRING_STEPS` without a closing quote, the buffer is returned as if
  complete. This is why two of the regex arguments are cut mid-pattern. It
  should raise instead, so a wrong value is never written.


## Testing strategy

Validation was driven by the evaluation criteria rather than by line
coverage.

**Input handling** — the CLI was run against a missing input file, a file
containing malformed JSON, and a file containing a JSON object instead of an
array. Each prints a clear one-line message and exits with code 1, with no
traceback:

```
Error: File not found: data/input/nope.json
Error: /tmp/bad.json is not valid JSON: Expecting property name enclosed in
double quotes: line 1 column 3 (char 2)
```

**Schema validation** — function definitions with an unsupported parameter
type, a missing `type` key, or unexpected extra keys are all rejected at
load time by `FunctionSchema`.

**Grammar boundaries** — the number predicates were checked by hand against
`""`, `"-"`, `"3"`, `"3."`, `".5"`, `"1.2.3"` and `"1e5"`; the enum
constraint against candidates that share a prefix and therefore cannot
resolve.

**Output contract** — the generated file is re-parsed with `json.load` and
each row checked for exactly the keys `prompt`, `name`, `parameters`, with
every parameter present in the schema and matching its declared type.

**Edge cases** — an empty function list, a prompt matching no function, and
generations that hit their step cap were all exercised to confirm the run
continues and reports rather than crashing.

**Static analysis** — `make lint` and `make lint-strict` both pass:
`flake8` reports nothing on `src/`, and `mypy --strict` type-checks all
eight modules cleanly.

## Visualization of the generation process

`src/monitor.py` traces decoding live, printing at each step how many tokens
the constraint allowed out of the full vocabulary, which token won, and the
value assembled so far. Colour is used to separate the field being generated
(yellow), the chosen token (green) and the running state (dim).

The trace makes the mechanism visible in a way the final JSON cannot: the
mask collapsing from six figures to a handful, and — in the example above —
`fn_add` narrowing to a single surviving candidate before the name is even
complete. From that point the remaining tokens are not really a choice at
all; the constraint has already decided.

## Resources

- Subject: *call me maybe — Introduction to function calling in LLMs*
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259) —
  the grammar the constraints implement
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Language Models are Unsupervised Multitask
  Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  — byte-level BPE, which explains the `Ġ` and `Ċ` markers in the
  vocabulary file
- https://www.intoai.pub/p/decoding-strategies-in-llms — decoding strategies

### Use of AI

AI (Claude) was used as a reviewer for:

- **Reviewing the decoding logic**, which surfaced the number-truncation
  bug, the missing escape handling in `StringConstraint`, and the silent
  truncation when a generation hits its step cap.
- **Profiling guidance** on the mask-scan cost and where the time actually
  goes.
- **Discussing prompt structure** for the argument-extraction step.
 