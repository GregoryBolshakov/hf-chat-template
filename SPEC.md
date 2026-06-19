# `hf-chat-template` — Technical Specification

What the crate does, what it guarantees, and how the compatibility layer is built. For exact
signatures see docs.rs; for the verified-model list see `COMPATIBILITY.md`.

- **Crate:** `hf-chat-template` · **Edition:** 2021 · **License:** MIT OR Apache-2.0
- **Engine:** `minijinja` 2.x plus a `transformers`-compatibility layer.

## 1. Purpose

Render a Hugging Face `chat_template` (the Jinja2 string in a model's `tokenizer_config.json`)
into the exact prompt string a model expects, given chat messages plus optional tools and
documents. The output is byte-for-byte identical to Python
`transformers.apply_chat_template(..., tokenize=False)`.

The engine already exists below us. The value of this crate is the thin compatibility layer plus
a golden test corpus that proves byte-identical output against the Python reference. Whitespace
and JSON key order are semantically load-bearing here: a stray newline or a re-sorted `tojson`
key silently corrupts every prompt downstream, which is why byte-identical testing is the core
deliverable.

## 2. Scope

In scope:

- Render `chat_template` strings in all three historical shapes (single string; named-template
  list; dict), including named sub-templates such as `tool_use` and `rag`.
- The standard input model: `messages` (role plus string-or-parts content), `tools`, `documents`,
  `add_generation_prompt`, and arbitrary extra template kwargs.
- The Jinja surface real templates use (section 5): `raise_exception`, `tojson`, `strftime_now`,
  Python string/list/dict methods, `namespace()` mutation, loop variables, whitespace control.
- Loading a template from a raw string, a parsed `tokenizer_config.json`, or a standalone
  `chat_template.jinja` file (the layout newer `transformers` use, where the template is not in
  `tokenizer_config.json`).
- Special-token substitution into the render context (section 8).

Optional, feature-gated (off by default):

- `hub`: fetch a model's template from the Hugging Face Hub (`from_hub`) — `tokenizer_config.json`
  plus a standalone `chat_template.jinja` when the repo ships one.
- `tokenizers`: render then encode to token IDs (`render_and_encode`).

Out of scope: inference, sampling, model loading, and authoring or editing templates. The crate
renders a string; turning it into token IDs is the caller's job (the `tokenizers` feature is a
convenience, not a core dependency).

## 3. Background: why this is hard

Real templates exercise corners a naive `minijinja::render` gets wrong or rejects:

1. **`raise_exception(msg)`** — templates call it to reject malformed conversations (role
   alternation, etc.). Must be a registered global that aborts the render carrying `msg`.
2. **`tojson`** with Python conventions — tool-calling templates serialize `tools` and tool-call
   arguments. Separators and key order must match Python, not Jinja defaults.
3. **`strftime_now(fmt)`** — newer templates (Llama-3.1+, Command-R) stamp the current date.
   Needs a clock, injectable for deterministic tests.
4. **Python methods on values** — `.strip()`, `.split()`, `.startswith()`, `dict.items()`, etc.
   Plain Jinja lacks these.
5. **`namespace()` cross-scope mutation** — `{% set ns = namespace(found=false) %}` then mutating
   `ns.found` inside loops, to carry state across iterations.
6. **Loop variables** — `loop.index0`, `loop.first`, `loop.last`, `loop.previtem`, `loop.nextitem`.
7. **Whitespace control** — `{%- … -%}`, `trim_blocks`, `lstrip_blocks`. Output whitespace is
   load-bearing.
8. **`add_generation_prompt`** — when true, the template appends the assistant turn's opening
   tokens. The crate passes the flag; the template does the work.
9. **Multimodal content** — `content` may be a string or a list of typed parts; templates branch
   on `content is string`.
10. **Tool-call round-trips** — assistant messages carry `tool_calls`; `tool`-role messages carry
    results. Formatting is highly model-specific.

The compatibility target is the `transformers` reference, including its globals and its Jinja
environment settings (`trim_blocks=True`, `lstrip_blocks=True`).

## 4. Dependencies & features

Mandatory:

- `minijinja` with `["json", "loop_controls", "preserve_order"]`. `preserve_order` is
  load-bearing: without it, Value maps sort keys and `tojson` diverges from Python on tool calls.
- `serde` (derive) for the input/config types.
- `serde_json` with `preserve_order` — the typed `render` path builds a `serde_json::Value`
  context, so its maps must stay insertion-ordered too. Also backs the custom `tojson`.
- `minijinja-contrib` (`pycompat`) — Python-method shim. Optional crate, pulled in by the
  default `pycompat` feature.

> **Order-preservation rule.** Data that may be `tojson`'d must keep insertion order through
> every layer. Two safe paths: deserialize directly into `minijinja::Value`, or route through
> `serde_json` with `preserve_order` (the typed path). The `context!` macro sorts keys and must
> not be used for order-sensitive data.

Features:

```toml
[features]
default = ["pycompat"]
pycompat   = ["dep:minijinja-contrib"]  # Python methods on values; opt out for minimal builds
hub        = ["dep:hf-hub"]             # from_hub / from_hub_revision
tokenizers = ["dep:tokenizers"]         # render_and_encode
```

`hf-hub` is pulled sync-only (ureq, rustls, no system OpenSSL). `tokenizers` is pulled
default-features-off with only the `onig` regex backend. The "just render a string" path stays
dependency-light so the crate is cheap to depend on.

## 5. Public API

`ChatTemplate` owns a compiled `minijinja::Environment`. Construction parses; `render` borrows
`&self`, so one instance is reusable and `Send + Sync` for server use. Exact signatures live on
docs.rs; this is the shape.

Construction:

- `ChatTemplate::from_str(&str)` and `impl FromStr` — from a raw template string.
- `ChatTemplate::from_tokenizer_config(&TokenizerConfig)` — resolves the default template and
  injects the config's special tokens.
- `ChatTemplate::from_template_and_config(&str, &TokenizerConfig)` — for the standalone-file
  layout: compile a `chat_template.jinja` string while taking special tokens from a separately
  loaded `tokenizer_config.json`. Matches the `transformers` precedence (the standalone file wins
  over any inline `chat_template` field), so the config's `chat_template` is ignored.
- `ChatTemplate::builder(&str)` / `builder_from_config(&TokenizerConfig)` — `ChatTemplateBuilder`
  for non-default options: `template_name`, `clock`, `pycompat`, `undefined_behavior`,
  `special_tokens_from(&TokenizerConfig)`, `build`.
- `ChatTemplate::from_hub(repo)` / `from_hub_revision(repo, rev)` — `hub` feature.

Rendering:

- `render(&RenderInput) -> Result<String, Error>` — the typed path; injects special tokens, then
  the input on top (input keys win).
- `render_messages(&[Message], add_generation_prompt) -> Result<String, Error>` — convenience.
- `render_value(minijinja::Value) -> Result<String, Error>` — escape hatch for contexts the typed
  model doesn't cover. Does not inject special tokens. Exposing `minijinja::Value` ties this entry
  point to minijinja's major version (section 9).
- `render_and_encode(&RenderInput, &tokenizers::Tokenizer) -> Result<(String, Vec<u32>), Error>` —
  `tokenizers` feature. Encodes with `add_special_tokens = false` because the template already
  emits the model's special tokens (matches `apply_chat_template(tokenize=True)`).

Input model (`serde`-(de)serializable, so request JSON deserializes straight in):

- `RenderInput { messages, tools, documents, add_generation_prompt, extra }`. `tools`,
  `documents`, and `extra` are `serde_json::Value` / `Map` — open-ended shapes that typing would
  constrain wrongly. `extra` is flattened into the top-level context.
- `Message { role, content, tool_calls, extra }`, with constructors `new(role, content)` and
  `system` / `user` / `assistant`. `tool_calls` and `extra` are raw JSON.
- `Content` is `#[serde(untagged)]` `Text(String)` or `Parts(Vec<Value>)`, preserving the
  string-vs-list distinction templates probe.

Config: `TokenizerConfig` exposes `chat_template` (a `ChatTemplateField`: single string or named
list) and the special tokens (`bos_token`, …, as a `TokenField`: string or object). Tolerant of
unknown fields.

Clock: the `Clock` trait supplies `strftime`. `FixedClock` (constructible from a unix timestamp
or y/m/d) pins dates for tests; `SystemClock` is a dependency-free default. `SystemClock` reads
the wall clock as UTC (section 7).

## 6. Error model

One `#[non_exhaustive] enum Error`:

- `TemplateRaised { message }` — the template called `raise_exception`; the input was rejected by
  the template's own validation. Distinct from engine errors so callers can act on it.
- `Compile(minijinja::Error)` — syntax error at construction.
- `Render(minijinja::Error)` — render-time error (undefined, type, unknown method).
- `Config(String)` — no usable `chat_template`, a missing named template, or a bad shape.
- `Hub(String)` — `hub` feature; fetch failure (network, auth, missing file).
- `Tokenize(String)` — `tokenizers` feature; encode failure.

Underlying `minijinja::Error` (line/span) is preserved via `source()`. The crate never panics on
user input; every fallible path returns `Result`. `Hub`/`Tokenize` carry strings so the upstream
error types stay out of the public API.

## 7. Jinja compatibility layer

Installed onto the `minijinja::Environment` to match the `transformers` reference:

- **Environment:** `trim_blocks` and `lstrip_blocks` on; `keep_trailing_newline` per the
  reference; auto-escape forced off (chat templates are not HTML).
- **`raise_exception(msg)`:** aborts the render via a sentinel-marked error recovered as
  `Error::TemplateRaised` (never reaches output).
- **`strftime_now(fmt)`:** formats the injected `Clock`. The dependency-free `SystemClock` formats
  UTC; `transformers` uses local time, so date-stamped templates differ unless a `FixedClock` or
  custom `Clock` is injected. The strftime implementation covers the specifiers real templates
  use; unknown specifiers pass through.
- **`tojson`:** a custom filter matching Python `json.dumps(x, ensure_ascii=False)` — `", "` /
  `": "` separators and insertion-ordered keys (minijinja's built-in sorts keys and omits spaces).
  Honors `tojson(indent=N)`.
- **`pycompat`:** `minijinja-contrib`'s unknown-method callback (`.strip()`, `.split()`,
  `.startswith()`, `| items`, …). Default feature; the supported set is documented in
  `COMPATIBILITY.md`.
- **Undefined behavior:** `UndefinedBehavior::Lenient` by default, reproducing the common
  `transformers` behavior. Overridable via the builder.
- **`namespace()` mutation:** supported by minijinja to the depth real templates need (verified by
  the corpus, including Hermes's macro/recursion path).

## 8. Template resolution & special tokens

`chat_template` appears as a single string or a named-template list. Resolution: an explicitly
requested name wins; else a `default` entry; else the sole template; else an ambiguity error
naming the available templates.

Newer `transformers` store the template in a standalone `chat_template.jinja` file rather than in
`tokenizer_config.json` (Gemma 3+, SmolLM3, LFM2, …). Such a file takes precedence over any inline
`chat_template` field, matching `transformers`. `from_template_and_config` is the offline entry
point (template string plus a config for the special tokens); `from_hub` applies this precedence
automatically when the repo ships the file. The `additional_chat_templates/` directory of named
standalone templates is not yet loaded.

Special tokens (`bos_token`, `eos_token`, …) are injected into the render context from
`TokenizerConfig`; a matching key in `RenderInput::extra` overrides. `add_generation_prompt` is
passed through as the boolean the template branches on; the crate never synthesizes the
generation prefix. No automatic BOS doubling: the crate renders exactly what the template emits
and never strips or adds special tokens silently. If a template emits `bos_token`, the caller
should set `add_special_tokens = false` at encode time.

## 9. The golden corpus

The corpus is the proof of the byte-identical claim and is treated as a first-class component.

Layout under `tests/corpus/<model-slug>/`:

```
meta.json              # model_id, revision (sha), license, source URL
tokenizer_config.json  # trimmed: chat_template (if inline) + special tokens
chat_template.jinja    # optional: the standalone template, for models that ship one
cases/<case>.json      # a RenderInput (optionally wrapped with a template_name)
expected/<case>.txt    # byte-exact reference output
```

When a model directory carries a `chat_template.jinja`, the runner loads the template from it and
takes special tokens from `tokenizer_config.json`, exercising the standalone-file path.

References are generated by `tools/gen_reference.py` (dev-only, not shipped), which runs the real
`transformers.apply_chat_template(..., tokenize=False)` at the pinned revision. The transformers
version is pinned and recorded; behavior changes are re-pinned deliberately. `tests/m3_corpus.rs`
renders each case and asserts byte-equality, printing a whitespace-visible diff on mismatch. CI
runs the corpus without Python. To add a model: drop in `meta.json` + trimmed
`tokenizer_config.json` + `cases/`, regenerate `expected/`, and commit.

## 10. Stability & versioning

SemVer. Pre-1.0 (`0.x`) while the corpus and API settle; `from_str`, `from_tokenizer_config`, and
`render` are treated as stable from `0.1`. The public surface is kept minimal — every exposed
type is a future compatibility obligation.

`minijinja` is re-exported (`pub use minijinja`) so downstreams construct `Value` without a
version-skew guess; `render_value` ties our API to minijinja's major version. With the
`tokenizers` feature, `tokenizers` is likewise re-exported and `render_and_encode` ties to its
version. The path to a stable `1.0` (including resolving these couplings, corpus breadth, and the
`strftime_now` divergence) is tracked outside this spec.

## 11. Documentation

Crate-level docs carry the quickstart, the boundary statement ("we emit a string; you tokenize"),
and the special-token/BOS caveat. `COMPATIBILITY.md` lists verified models, supported Python
methods, and known divergences; it is the living record of the moat and is kept current.
