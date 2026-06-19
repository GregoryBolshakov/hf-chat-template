# Compatibility

"Compatible" here means one thing precisely: for a given `(chat_template, input)`, this crate
produces **the same bytes** as Python `transformers.apply_chat_template(..., tokenize=False)`.
That claim is enforced by `tests/m3_corpus.rs`, which renders real model templates and diffs
against committed reference output from `transformers` (see `tests/corpus/README.md` for the
pinned version and provenance).

## Verified models

Byte-identical in CI:

| Model | Cases exercised |
|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | basic, no-system, single-user, tool calling |
| Qwen/Qwen3-0.6B | basic, no-system, single-user, tool calling |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | basic, no-system, single-user |
| microsoft/Phi-3-mini-4k-instruct | basic, no-system, single-user |
| NousResearch/Hermes-3-Llama-3.1-8B | basic, no-system, single-user, named `tool_use` template |
| LiquidAI/LFM2-1.2B | basic, no-system, single-user, tool list (standalone `chat_template.jinja`) |
| HuggingFaceTB/SmolLM3-3B | standalone `chat_template.jinja`, `{% generation %}` block (reasoning) |

This is the v1 corpus (ungated models). LFM2 ships its template as a standalone
`chat_template.jinja` file rather than inline in `tokenizer_config.json`, exercising that loading
path. Expanding toward the major gated families (Llama-3.x, Gemma, Mistral) is in progress and
requires a Hugging Face token to fetch.

## Jinja surface supported

Confirmed against real templates and the corpus:

- **Control flow & whitespace**: `trim_blocks`, `lstrip_blocks`, `keep_trailing_newline`, and
  `{%- … -%}` whitespace control, matching the `transformers` Jinja environment.
- **`namespace()` cross-scope mutation** — `{% set ns = namespace(found=false) %}` then mutating
  `ns.found` inside loops.
- **Macros & recursion** (e.g. Hermes's `tool_use` template).
- **Loop variables**: `loop.index0`, `loop.last`, `loop.first`, `loop.previtem`, `loop.nextitem`.
- **`raise_exception(msg)`** — surfaces as a distinct [`Error::TemplateRaised`], not an engine
  error, so you can tell "the template rejected this conversation" from "the library has a bug."
- **`strftime_now(fmt)`** — via an injectable [`Clock`] (see caveat below).
- **`tojson`** — a custom filter matching Python `json.dumps(x, ensure_ascii=False)`: `", "` /
  `": "` separators and **insertion-ordered** keys (minijinja's built-in sorts keys and omits
  spaces). Honors `tojson(indent=N)`.
- **`pycompat`** (default feature) — Python methods on values: `.strip()`, `.lstrip()`,
  `.rstrip()`, `.split()`, `.startswith()`, `.endswith()`, `.title()`, `.upper()`, `.lower()`,
  `| items`, `| trim`, and more, provided by `minijinja-contrib`.
- **`{% generation %}` block** — the `transformers` AssistantTracker block (marks assistant token
  spans for `return_assistant_tokens_mask`). It is rewritten to an output-neutral `{% if true %}`
  before compilation, so reasoning templates that use it (SmolLM3) render byte-identically. The
  span metadata itself is not exposed; this crate emits a string, not a token mask.

## Known divergences & caveats

- **`strftime_now` defaults to UTC.** `SystemClock` reads `SystemTime` as UTC; `transformers`
  uses Python local time. Templates that stamp the date (Llama-3.1, Command-R) will differ by
  timezone unless you inject a `FixedClock`/custom `Clock`. The corpus deliberately pins dates
  for any such model.
- **Undefined variables are lenient by default** (`UndefinedBehavior::Lenient`), matching the
  common `transformers` behavior. Override via the builder if a template needs strict semantics.
- **`pycompat` coverage is `minijinja-contrib`'s set.** An exotic Python method a template relies
  on may be missing; if you hit one, it shows up as a render error, not silent wrong output.
- **No automatic BOS doubling.** If a template emits `bos_token`, set `add_special_tokens = false`
  at encode time. This crate never strips or adds special tokens behind your back.

## Reporting a mismatch

If you find a template that renders differently from `transformers`, that's a bug worth a report.
Ideally include the `chat_template`, the input messages, and both outputs. Adding it to the corpus
(`tests/corpus/`) and regenerating references with `tools/gen_reference.py` turns the fix into a
permanent regression test.

[`Error::TemplateRaised`]: https://docs.rs/hf-chat-template/latest/hf_chat_template/enum.Error.html
[`Clock`]: https://docs.rs/hf-chat-template/latest/hf_chat_template/trait.Clock.html
