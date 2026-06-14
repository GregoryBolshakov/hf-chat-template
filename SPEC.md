# `hf-chat-template` — Technical Specification

**Status:** Draft v1 (implementation-ready)
**Crate name:** `hf-chat-template` (verified available on crates.io 2026-06-13)
**Edition:** 2021 · **MSRV:** 1.74 (tentative; pin to whatever `minijinja` requires — verify)
**License:** MIT OR Apache-2.0 (dual, ecosystem-standard)

---

## 1. Purpose and thesis

Render a Hugging Face **`chat_template`** (the Jinja2 string embedded in a model's
`tokenizer_config.json`) into the exact prompt string a model expects, given a list of
chat messages plus optional tools/documents. This is the layer every Rust local-inference
stack (candle, mistral.rs, burn-based servers, llama.cpp-alternatives) currently
reimplements ad-hoc or vendors badly.

**The product is correctness, not features.** The engine (`minijinja`) already exists below
us with 22.5M downloads. Our value is the thin compatibility shim plus a **golden test
corpus** proving we render the top ~50 models' templates byte-identically to the Python
reference (`transformers.PreTrainedTokenizer.apply_chat_template`). That corpus is the moat:
a weekend clone cannot match it, and getting this subtly wrong silently corrupts every prompt
downstream.

### Success criteria
- Byte-identical output vs. the Python `transformers` reference for every model in the corpus.
- Zero required transitive deps beyond `minijinja` (+ `serde`); everything else feature-gated.
- API stable enough that downstream crates depend on it rather than vendoring — this is the
  whole point (download growth comes from being a transitive dependency, not an app).

---

## 2. Scope

### In scope
- Parse + render HF `chat_template` strings (single template **and** the named-template dict
  form `{"default": "...", "tool_use": "...", "rag": "..."}`).
- The standard input model: `messages` (role/content + multimodal content lists), `tools`
  (JSON-schema function defs), `documents` (RAG), `add_generation_prompt`, and arbitrary
  extra template kwargs.
- The Jinja compatibility surface HF templates actually use (see §6): `raise_exception`,
  `tojson`, `strftime_now`, Python string/list/dict methods, `namespace()`, loop vars.
- Loading the template from: a raw string, a `tokenizer_config.json` value, or a file path.
- Special-token substitution policy (bos/eos handling — see §9).

### Out of scope (v1)
- **Tokenization.** We emit a *string*. Turning it into token IDs is the caller's job
  (`tokenizers`, `tiktoken-rs`, etc.). Keeping this boundary is what makes us a small
  depended-upon library rather than a heavy framework. A `tokenizers`-integration helper may
  come later behind a feature flag, never as a core dep.
- Downloading `tokenizer_config.json` from the Hub. That's `hf-hub`'s job; we accept its
  output. (Optional `hf-hub` convenience constructor may be feature-gated later.)
- Inference, sampling, model loading.
- Authoring/editing templates. We render, we don't generate templates.

---

## 3. Background: why this is hard (the quirks that justify the corpus)

HF chat templates are Jinja2, but real templates in the wild exercise corners that a naive
`minijinja::render` will get wrong or reject:

1. **`raise_exception(msg)`** — a global injected by `transformers`. Templates call it to
   reject malformed conversations (e.g. alternating-role enforcement). Must be a registered
   global that produces a render error carrying `msg`.
2. **`tojson`** filter with HF's argument conventions (`indent=`, ensuring ASCII behavior).
   Tool-calling templates serialize `tools` and tool-call arguments with `| tojson`.
3. **`strftime_now(format)`** — newer templates (Llama-3.1+, Command-R) inject the current
   date for system prompts. Needs a clock (injectable for deterministic tests).
4. **Python string/list/dict methods** — templates call `.strip()`, `.split()`,
   `.startswith()`, `.endswith()`, `.title()`, `.rstrip()`, `.lstrip()`, `.replace()`,
   `dict.items()`, `list` indexing, etc. Plain Jinja doesn't have these; `minijinja` needs
   the `pycompat` unknown-method callback.
5. **`namespace()`** + cross-scope mutation `{% set ns.found = true %}` — used heavily to
   carry state across `{% for %}` iterations (Gemma, many tool templates). Verify minijinja
   support depth (see §15 open questions).
6. **Loop variables** — `loop.index0`, `loop.first`, `loop.last`, `loop.length`.
7. **Whitespace control** — `{%- … -%}`. Output whitespace is *semantically load-bearing*
   here: a stray newline changes the prompt. This is precisely why byte-identical testing
   matters.
8. **`add_generation_prompt`** — when true, append the assistant turn's opening tokens
   (e.g. `<|im_start|>assistant\n`) so generation continues correctly.
9. **Multimodal content** — `message["content"]` may be a string *or* a list of
   `{"type": "text"|"image"|..., ...}` parts. Templates branch on `content is string`.
10. **Tool-call round-trips** — assistant messages carry `tool_calls`; `tool` role messages
    carry results. Template formatting of these is highly model-specific.

The Python reference behavior is defined in `transformers` (`apply_chat_template`,
`jinja_utils`); our compatibility target is *that*, including its specific globals and its
Jinja environment settings (`trim_blocks=True`, `lstrip_blocks=True` — **verify exact
defaults** against the transformers source; these flags change output whitespace and must
match).

---

## 4. Dependencies & feature flags

### Mandatory
- `minijinja` with features **`["json", "loop_controls", "preserve_order"]`** (pinned in M1).
  `preserve_order` is **load-bearing**: without it, Value maps sort keys and `tojson` output
  diverges from Python for tool-calling. See `M0_FINDINGS.md` and the tojson note below.
- `serde` (derive) — message/tool input types and arbitrary-value plumbing.
- `serde_json` with **`preserve_order`** — the typed `render()` path (M2) builds a
  `serde_json::Value` context, so its object maps must stay insertion-ordered too, else
  tool-call `tojson` output sorts keys. (In M1 alone this was unnecessary; M2's typed model
  made it load-bearing.) Also used by our custom `tojson` serializer.
- `minijinja-contrib` with `pycompat` — Python method shim.

> **Order-preservation rule (important):** template data that may be `tojson`'d must keep
> insertion order through *every* layer it passes. Two safe paths: (a) deserialize directly
> into `minijinja::Value` (`serde_json::from_str::<Value>`); (b) route through `serde_json`
> with `preserve_order` on (the M2 typed path). The `context!` macro sorts keys regardless and
> must not be used for order-sensitive data.

### Optional (feature-gated, off by default)
- `minijinja-contrib` (feature `pycompat`) → gated under our `pycompat` feature
  (**default-on**, because real templates need it; but allow opt-out for minimal builds).
- `time` or `chrono` → gated under `strftime` feature for `strftime_now` real-clock impl.
  Default impl can be a no-extra-dep fixed/injected clock; real wall-clock needs this.
- `serde_json` → gated under `json-value` for ergonomic `serde_json::Value` input/round-trip.
- `hf-hub` → gated under `hub`, a convenience constructor `from_hub(repo)`. Never a core dep.
- `tokenizers` → gated under `tokenizers`, a helper to also return token IDs. Future.

### Proposed `Cargo.toml` feature table
```toml
[features]
default = ["pycompat"]
pycompat = ["dep:minijinja-contrib"]
strftime = ["dep:time"]
json-value = ["dep:serde_json"]
hub = ["dep:hf-hub"]
tokenizers = ["dep:tokenizers"]
```

Rationale: keep the dependency-light "just render a string" path the default so we're cheap
to pull in (low friction = more dependents = more downloads).

---

## 5. Public API

Design goals: small, hard to misuse, `&self` render so an instance is reusable and `Send +
Sync` for server use. Anchor for the C++ reader: `ChatTemplate` owns a compiled
`minijinja::Environment` (like a compiled regex / prepared statement); construction does the
parse, `render` is cheap and immutable-borrow so it's shareable across threads without a
mutex.

### 5.1 Core type

```rust
/// A compiled Hugging Face chat template, ready to render.
///
/// Construction compiles the Jinja source and installs the HF-compat globals/filters.
/// `render` borrows `&self`, so one instance is shareable across threads.
pub struct ChatTemplate {
    // owns a minijinja::Environment<'static> with the template added under a fixed name,
    // plus config (clock, options).
}

impl ChatTemplate {
    /// Compile from a raw Jinja chat-template string.
    pub fn from_str(source: &str) -> Result<Self, Error>;

    /// Compile from a parsed `tokenizer_config.json` value, reading the `chat_template`
    /// field. Supports both the string form and the list-of-named-templates form
    /// `[{"name": "...", "template": "..."}]` / dict form.
    pub fn from_tokenizer_config(config: &TokenizerConfig) -> Result<Self, Error>;

    /// Builder entry point for non-default options (clock, strict mode, etc.).
    pub fn builder(source: &str) -> ChatTemplateBuilder;

    /// Render the template to the final prompt string.
    pub fn render(&self, input: &RenderInput) -> Result<String, Error>;

    /// Convenience: render with just messages + add_generation_prompt, no tools/docs.
    pub fn render_messages(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, Error>;
}
```

> **Naming note:** prefer an inherent `from_str` *and* implement `std::str::FromStr` so both
> `ChatTemplate::from_str(s)` and `s.parse::<ChatTemplate>()` work. (C++ intuition check:
> unlike a converting constructor, `FromStr` is the idiomatic fallible-parse trait; provide
> it.)

### 5.2 Builder

```rust
pub struct ChatTemplateBuilder { /* ... */ }

impl ChatTemplateBuilder {
    /// Inject a deterministic clock (for `strftime_now`) — essential for golden tests.
    pub fn clock(self, clock: impl Clock + 'static) -> Self;

    /// Enable/disable the pycompat method shim (default: enabled when feature on).
    pub fn pycompat(self, enabled: bool) -> Self;

    /// How to treat undefined variables. Default mirrors transformers (see §6.5).
    pub fn undefined_behavior(self, ub: UndefinedMode) -> Self;

    /// Select a named sub-template when the config carries several (e.g. "tool_use").
    pub fn template_name(self, name: &str) -> Self;

    pub fn build(self) -> Result<ChatTemplate, Error>;
}
```

### 5.3 Input data model

These mirror the OpenAI/HF message shape. `Message.content` is an enum to handle the
string-or-parts multimodal reality. Everything is `serde`-(de)serializable so callers can
deserialize their existing JSON straight in.

```rust
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RenderInput {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<Tool>,            // JSON-schema function definitions
    #[serde(default)]
    pub documents: Vec<Document>,    // RAG documents
    #[serde(default)]
    pub add_generation_prompt: bool,
    /// Arbitrary extra kwargs some templates read (e.g. `enable_thinking`,
    /// `builtin_tools`). Passed through to the Jinja context verbatim.
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>, // gated: see note
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: String,                 // "system" | "user" | "assistant" | "tool" | ...
    #[serde(default)]
    pub content: Content,             // string OR list of parts (see below)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>, // name, tool_call_id, etc.
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),          // [{type:"text",text:...},{type:"image",...}]
}
impl Default for Content { fn default() -> Self { Content::Text(String::new()) } }
```

> **Design tension — typed vs. transparent.** Templates read arbitrary keys we can't fully
> enumerate (every model invents fields). Two options:
>
> - **(A) Strongly-typed structs** (above) with `#[serde(flatten)] extra` escape hatches.
>   Ergonomic, documents the common path, but the `flatten` + `serde_json::Value` pulls
>   `serde_json` into the core. 
> - **(B) Fully transparent:** accept `&[serde_json::Value]` / a single
>   `minijinja::Value` context and do zero modeling.
>
> **Decision:** ship **both**. The typed API (A) is the documented happy path; expose a
> lower-level `render_value(&self, ctx: minijinja::Value) -> Result<String, Error>` that
> takes an arbitrary context for callers who need fields we didn't model. The typed structs
> serialize *into* a `minijinja::Value` internally, so there's one render path. Put the
> `serde_json::Value` extras behind `json-value` (default-on) so the minimal build can drop
> it by using only `render_value`.

### 5.4 `Clock` trait (for `strftime_now`)

```rust
/// Supplies "now" to `strftime_now`. Injectable so golden tests are deterministic.
pub trait Clock: Send + Sync {
    /// Return current local datetime formatted per the given strftime format string.
    fn strftime(&self, format: &str) -> String;
}

/// Always returns a fixed instant — used in tests and reproducible builds.
pub struct FixedClock(pub /* some datetime repr */);

/// Real wall clock (feature = "strftime").
#[cfg(feature = "strftime")]
pub struct SystemClock;
```

> Keeping the clock behind a trait (rather than calling `SystemClock` directly) is what lets
> the golden corpus pin dates and stay byte-stable. This is load-bearing for the moat.

---

## 6. Jinja compatibility layer

This section defines exactly what we install onto the `minijinja::Environment`. **Target =
the `transformers` reference environment.** Each item below must be cross-checked against
the transformers source (`src/transformers/utils/chat_template_utils.py` / `jinja`-related
code) AND against minijinja's actual behavior with a unit test.

### 6.1 Environment settings
- `trim_blocks` and `lstrip_blocks`: set to **match transformers' Jinja env** (transformers
  uses a Jinja `Environment` with specific block-trim settings — **VERIFY exact values**;
  they directly change whitespace and therefore byte-equality). minijinja exposes these via
  `Environment::set_trim_blocks` / `set_lstrip_blocks` (verify method names).
- Keep `minijinja` default auto-escape **off** for our template name (chat templates are not
  HTML; escaping would corrupt output). Explicitly set an auto-escape callback returning
  `AutoEscape::None` for our template.

### 6.2 Globals
- `raise_exception(message)` → registered via `Environment::add_function`. Implementation
  returns `Err` of a minijinja error whose detail carries `message`; surfaces as
  `Error::TemplateRaised { message }` (see §7).
- `strftime_now(format)` → calls the injected `Clock`.
- Possibly `now()` / others transformers injects — **enumerate from transformers source and
  match the full set**; do not invent extras.

### 6.3 Filters
- `tojson` — provided by minijinja's `json` feature. **Verify** it matches Python
  `tojson`'s defaults transformers relies on (separators, `ensure_ascii`, `indent`
  handling). If minijinja's differs, register our own `tojson` that matches Python output
  byte-for-byte. This is a likely divergence point — budget time for it.

### 6.4 Python-method compatibility (`pycompat`)
- Register `minijinja_contrib::pycompat::unknown_method_callback` via
  `Environment::set_unknown_method_callback`.
  **VERIFY exact path + signature** on docs.rs/minijinja-contrib before coding; the function
  exists ("An unknown method callback implementing python methods on primitives") but the
  precise signature and registration call must be confirmed.
- Audit coverage: collect every `.method()` call across the corpus templates and assert
  pycompat implements each (`strip`, `lstrip`, `rstrip`, `split`, `rsplit`, `startswith`,
  `endswith`, `title`, `upper`, `lower`, `replace`, `items`, `keys`, `values`, `get`,
  `join`...). For any method pycompat lacks, register a supplemental method/filter. Document
  the supported set explicitly.

### 6.5 Undefined behavior
- transformers' Jinja raises on undefined in some configs and is lenient in others.
  **Determine the reference behavior** and map to `minijinja::UndefinedBehavior`
  (`Lenient` / `Chainable` / `Strict`). Expose override via builder (`UndefinedMode`).
  Default should reproduce reference output for the corpus.

### 6.6 `namespace()` and cross-scope mutation
- Required by many templates (`{% set ns = namespace(x=0) %}` then `{% set ns.x = ... %}`
  inside loops). **VERIFY minijinja supports namespace mutation to the depth templates need.**
  If it does not, this is the single biggest compatibility risk and may require either a
  minijinja version bump, an upstream contribution, or a template-level shim. Flag early;
  test first (see §15).

---

## 7. Error model

One `enum Error` (with `thiserror`-style display, but consider hand-rolling to avoid the dep
— `thiserror` is cheap and ubiquitous, acceptable). Variants:

```rust
pub enum Error {
    /// The template called raise_exception(msg) — a deliberate rejection of the input.
    TemplateRaised { message: String },
    /// Jinja compile error (syntax) at construction.
    Compile { source: minijinja::Error },
    /// Jinja render error (undefined, type error, unknown method, ...).
    Render { source: minijinja::Error },
    /// tokenizer_config.json had no chat_template / unknown named template / bad shape.
    Config(ConfigError),
    /// Caller asked for a named template that doesn't exist.
    UnknownTemplate { name: String },
}
```

- `TemplateRaised` must be cleanly distinguishable from engine errors — callers act on it
  (reject the conversation) vs. log a bug. This separation is a real ergonomic win and worth
  the variant.
- Preserve the underlying `minijinja::Error` (line numbers, template span) via `source()`.
- Implement `std::error::Error` + `Display`. No panics on any user input — malformed input
  is always a `Result::Err`. (C++ check: there are no exceptions; every fallible path is in
  the return type. A panic would be a library bug, treated as such.)

---

## 8. Template source resolution

`tokenizer_config.json`'s `chat_template` appears in three historical shapes; support all:
1. **String:** `"chat_template": "{% for ... %}"`.
2. **List of named dicts:** `[{"name":"default","template":"..."},
   {"name":"tool_use","template":"..."}]`.
3. (Legacy/edge) some configs put it in `chat_template.jinja` files or under other keys —
   accept an explicit string path too.

Resolution rules:
- If a name is requested via builder, use it; error `UnknownTemplate` if absent.
- Else if a `"default"` named template exists, use it.
- Else if exactly one template exists, use it.
- Else error (ambiguous), telling the caller which names exist.

`TokenizerConfig` is a thin serde struct exposing at least `chat_template` plus the special
tokens we may need for §9 (`bos_token`, `eos_token`, etc.), tolerant of unknown fields
(`#[serde(default)]`, no `deny_unknown_fields`).

---

## 9. Special tokens & generation prompt

- Templates frequently reference `bos_token`, `eos_token` as **context variables**
  (`{{ bos_token }}`), not literals. transformers injects these from the tokenizer config.
  We must populate them into the render context from `TokenizerConfig` (or let the caller
  pass them via `extra`). **Document precisely** which special-token variables we inject and
  from where; mismatches here are a classic silent-corruption bug.
- `add_generation_prompt`: pass through to the context as the boolean the template branches
  on. We do **not** synthesize the generation prefix ourselves — the template does. Our job
  is only to pass the flag faithfully.
- **No automatic BOS doubling:** a frequent real bug is the template emitting `bos_token`
  *and* the tokenizer adding BOS again at encode time. We render exactly what the template
  says and document that the caller must set `add_special_tokens=false` at encode time if the
  template already emits BOS (mirrors the transformers guidance). We do not silently strip.

---

## 10. The golden test corpus (the moat)

This is the deliverable that creates defensibility and trust. Treated as a first-class
component, not an afterthought.

### 10.1 Structure
```
tests/corpus/
  <model-slug>/
    chat_template.jinja        # extracted template source
    config.json                # special tokens, trim/lstrip flags actually used
    cases/
      basic.json               # {input, expected_output}
      tool_use.json
      multimodal.json
      system_prompt.json
      ...
    LICENSE / PROVENANCE.md     # where the template came from + its license
```

### 10.2 Reference generation
- A non-shipped Python helper script (`tools/gen_reference.py`, dev-only, documented but not
  part of the crate) runs `transformers.apply_chat_template` for each `(template, input)`
  and writes `expected_output`. This pins us to ground truth.
- Pin the `transformers` version used to generate references; record it in `PROVENANCE.md`.
  When transformers changes behavior, we re-pin deliberately.

### 10.3 Coverage targets (v1 corpus = top ~15–20 models, grow to 50)
Prioritize by ecosystem weight and template complexity:
- Llama-3 / 3.1 / 3.2 (uses `strftime_now`, tool calls, `bos_token`)
- Qwen2.5 / Qwen3 (`<|im_start|>`, thinking flags)
- Mistral / Mixtral (no system role quirk, `[INST]`)
- Gemma 2 / 3 (`namespace()`, multimodal)
- Phi-3 / Phi-4
- Command-R / R+ (RAG `documents`, grounded generation — heavy template)
- DeepSeek
- ChatML baseline
- A tool-calling-heavy one (Hermes / functionary-style)

### 10.4 Test harness
- A single Rust integration test iterates the corpus, renders with `hf-chat-template`,
  asserts **byte-equality** with `expected_output`. Diffs print a char-level mismatch with
  context (whitespace made visible) because whitespace bugs dominate.
- CI fails on any mismatch. A "known-divergence" allowlist (with a reason + tracking issue)
  is permitted but must be explicit and small.

---

## 11. Performance

- Construction (compile) is the costly step; `render` should be allocation-light. Reuse one
  `ChatTemplate` across requests (it's `&self`-render and `Sync`).
- Target: render of a typical 10-message conversation in well under 100µs after warm compile
  (sanity bound; benchmark with `criterion` in a dev-only bench, not a dep).
- Avoid cloning the template source per render; the `Environment` holds it once.
- `RenderInput → minijinja::Value` conversion: stream via `serde` rather than building an
  intermediate `serde_json::Value` when on the typed path, if it measures faster. Measure
  before optimizing.

---

## 12. API stability & versioning

- SemVer. Pre-1.0 (`0.x`) while the corpus and pycompat coverage settle, but treat the core
  three methods (`from_str`, `from_tokenizer_config`, `render`) as stable from `0.1`.
- Keep the public surface minimal; every type we expose is a future compat obligation. Hide
  `minijinja` types from the signature where reasonable **except** `render_value` (which
  deliberately exposes `minijinja::Value` as the escape hatch — document that this ties our
  public API to a `minijinja` major version, and re-export the version we use).
- Re-export `pub use minijinja;` so downstreams can construct `Value` without a version-skew
  guess.

---

## 13. Documentation

- Crate-level doc: the 10-line quickstart (load config → render), the boundary statement
  ("we emit a string; you tokenize"), and the special-token/BOS caveat.
- A `COMPATIBILITY.md` listing supported models (corpus), supported Python methods, and known
  divergences — this doc *is* marketing for the moat; keep it current.
- Every public item documented; `#![deny(missing_docs)]`.

---

## 14. Testing strategy (beyond the corpus)

- Unit tests per compat shim: `raise_exception` surfaces as `TemplateRaised`; `tojson`
  matches Python output on tricky values (unicode, nested, `indent`); each pycompat method.
- Property/fuzz-lite: malformed inputs never panic (assert `Err`, never unwind). Consider
  `cargo fuzz` target on `from_str` + `render` later.
- `trybuild`-style or doc-tests for the public examples.
- MSRV CI job; `cargo +nightly` doc job; `clippy -D warnings`; `cargo fmt --check`.
- `cargo deny` for license/dup-dep hygiene (helps trust; many downstreams vet this).

---

## 15. Open questions / verification items

**M0 status (resolved 2026-06-13 against minijinja 2.20.0 — see `tests/m0_spikes.rs` and
`M0_FINDINGS.md`):**

1. **`namespace()` mutation depth — ✅ RESOLVED, WORKS.** Both accumulator and bool-flag
   patterns render correctly. *This was the go/no-go gate; minijinja 2.20 passes. GO.*
2. **transformers' exact Jinja env flags — ⏳ partially.** `set_trim_blocks`/`set_lstrip_blocks`
   confirmed to exist and compile; **exact transformers values still to be read from source**
   and pinned via the corpus. (transformers uses `trim_blocks=True, lstrip_blocks=True` — verify.)
3. **pycompat — ✅ RESOLVED.** Exact API: `env.set_unknown_method_callback(
   minijinja_contrib::pycompat::unknown_method_callback)`. `strip/split/startswith/title/
   endswith` all work.
4. **`tojson` parity — ✅ DIVERGENCE CONFIRMED; needs custom impl.** minijinja emits
   **sorted keys, no spaces** (`{"a":"x","b":1}`). Python/transformers `json.dumps(
   ensure_ascii=False)` emits **insertion order, spaces** (`{"b": 1, "a": "x"}`). **M1 must
   register a custom `tojson`** matching Python: `ensure_ascii=False`, separators `(", ",
   ": ")` at indent 0, insertion-order preservation, and HF's `indent=` handling.
5. **Full global set transformers injects** (`raise_exception`, `strftime_now`, anything
   else). Enumerate from source; match exactly; invent nothing. *(raise_exception mechanism
   ✅ proven in spike.)*
6. **minijinja method names — ✅ RESOLVED.** `add_function`, `set_trim_blocks`,
   `set_lstrip_blocks`, `set_unknown_method_callback` all confirmed.
7. **MSRV** — set to the max of our needs and `minijinja`'s declared MSRV. *(toolchain here is
   1.95; pin conservatively for downstreams.)*
8. **Special-token variable set** transformers injects into the context (beyond
   `bos_token`/`eos_token`?). Still to enumerate from source.

Remaining before/within M1: items **2, 4 (impl), 5, 8** — all now concrete tasks, none a
design risk.

---

## 16. Implementation milestones

- **M0 — Spikes (de-risk):** ✅ **DONE 2026-06-13.** All 8 spikes pass on minijinja 2.20.0.
  Gate passed: `namespace()` mutation works. Engine choice confirmed. See `M0_FINDINGS.md`.
- **M1 — Core render path:** ✅ **DONE 2026-06-13.** `from_str`/`builder`/`render_value`,
  `FromStr`, the compat globals (`raise_exception`→`TemplateRaised`, custom Python `tojson`,
  `strftime_now` via dependency-free `Clock`/`FixedClock`/`SystemClock`), pycompat wired,
  whitespace flags, error model. ChatML + Mistral render byte-correctly. 18 tests green,
  clippy clean. Modules: `error`, `clock`, `json`, `engine`, `template`, `lib`.
- **M2 — Typed input model + config loading:** ✅ **DONE 2026-06-14.** `RenderInput`/`Message`/
  `Content` (untagged string-or-parts), `TokenizerConfig` + three `chat_template` shapes,
  `from_tokenizer_config`/`builder_from_config`, named-template resolution (§8), special-token
  injection (§9, input overrides), `render`/`render_messages`. `serde_json` `preserve_order`
  enabled. 11 tests in `tests/m2_model.rs`. Deviation from §5.3: arbitrary holes (`tools`,
  `tool_calls`, content parts, `extra`) stay `serde_json::Value` rather than fully-typed
  `Tool`/`ToolCall`/`Document` structs — open-ended shapes that typing would only constrain
  wrongly; feature-gating of the `serde_json::Value` extras deferred to M4.
- **M3 — Corpus v1:** ✅ **Foundation DONE 2026-06-14.** `tools/gen_reference.py` (real
  `transformers` 5.12.0), corpus harness (`tests/m3_corpus.rs`), `.gitattributes` (byte-exact
  refs), CI (`.github/workflows/ci.yml`: fmt + clippy `-D warnings` + tests). **5 ungated models,
  18 cases byte-identical** to the reference — Qwen2.5, Qwen3, SmolLM2, Phi-3, Hermes-3-Llama-3.1
  — including tool-calling (`tojson` key-order) and Hermes's named `tool_use` sub-template with
  Jinja macros + recursion. **Remaining for full M3:** expand to the SPEC's 15–20 (the rest —
  Llama-3.x, Gemma, Mistral — are **gated**; need an HF token to fetch). Date-pinned
  (`strftime_now`) models deferred to M5.
- **M4 — Polish & publish:** docs, `COMPATIBILITY.md`, feature-flag hygiene, `0.1.0` to
  crates.io. `pub use minijinja`.
- **M5 — Growth:** expand corpus toward 50 models; optional `hub` + `tokenizers` features;
  announce in the candle/mistral.rs/llama-cpp-rs orbits to seed adoption (downloads come from
  becoming a dependency).

---

## 17. Risks & mitigations (summary)

| Risk | Severity | Mitigation |
|---|---|---|
| minijinja can't do `namespace()` mutation | High | M0 spike first; upstream PR or shim; worst case fork engine choice |
| `tojson` / whitespace byte-mismatches | Med | Golden corpus catches them; custom `tojson` if needed |
| pycompat missing a method some template uses | Med | Corpus audit of all `.method()` calls; register supplements |
| transformers changes reference behavior | Low-Med | Pin transformers version in PROVENANCE; deliberate re-pin |
| Big SDKs vendor their own instead of depending | Med (download ceiling) | Make it trivially cheap + obviously more correct (corpus) than vendoring |
| API ties to minijinja major version via `render_value` | Low | Re-export minijinja; document; bump our major with theirs |

---

*End of specification. Authored as the implementation contract for `hf-chat-template`.
Every "VERIFY" item is load-bearing: the compiler and the `transformers` source are ground
truth and override any assumption written here.*
