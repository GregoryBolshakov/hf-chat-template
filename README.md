# hf-chat-template

[![CI](https://github.com/GregoryBolshakov/hf-chat-template/actions/workflows/ci.yml/badge.svg)](https://github.com/GregoryBolshakov/hf-chat-template/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/hf-chat-template.svg)](https://crates.io/crates/hf-chat-template)
[![docs.rs](https://img.shields.io/docsrs/hf-chat-template)](https://docs.rs/hf-chat-template)
[![license](https://img.shields.io/crates/l/hf-chat-template.svg)](#license)

Render a Hugging Face **`chat_template`** — the Jinja2 string embedded in a model's
`tokenizer_config.json` — into the exact prompt a model expects, **byte-for-byte identical** to
Python's `transformers.apply_chat_template`.

If you do local inference in Rust (candle, mistral.rs, llama-cpp bindings, a custom server), this
is the prompt-building layer you would otherwise reimplement by hand — and get subtly wrong. A
single stray newline or a re-sorted `tojson` key silently corrupts every prompt downstream. This
crate's job is to *not* do that, and to prove it against a golden corpus on every commit.

```toml
[dependencies]
hf-chat-template = "0.1"
```

## Example

```rust
use hf_chat_template::{ChatTemplate, Message};

let tmpl = ChatTemplate::from_str(
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
     {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
)?;

let prompt = tmpl.render_messages(&[Message::new("user", "Hello!")], /* add_generation_prompt = */ true)?;
assert_eq!(prompt, "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n");
# Ok::<(), hf_chat_template::Error>(())
```

Or load a real model's config directly — special tokens (`bos_token`, `eos_token`, …) are
injected for you, and the named-template forms (`tool_use`, `rag`) are resolved:

```rust,no_run
use hf_chat_template::{ChatTemplate, Message, TokenizerConfig};

let json = std::fs::read_to_string("tokenizer_config.json")?;
let cfg: TokenizerConfig = serde_json::from_str(&json)?;
let tmpl = ChatTemplate::from_tokenizer_config(&cfg)?;

let prompt = tmpl.render_messages(&[Message::new("user", "Hi")], true)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

For full control — tools, documents, or model-specific kwargs the typed model doesn't
name — build a [`RenderInput`] (or drop to `render_value` with an arbitrary `minijinja::Value`).

## What it does

- **Correctness is the product.** The engine ([`minijinja`](https://crates.io/crates/minijinja))
  already exists below us. The value here is the thin `transformers`-compatibility layer plus a
  corpus proving byte-identical output on real models.
- Installs the globals templates actually use: `raise_exception`, `strftime_now`, a
  **Python-compatible `tojson`** (matching `json.dumps(…, ensure_ascii=False)` separators *and*
  key order), and Python string/list/dict methods via `pycompat`.
- Handles all three `chat_template` shapes (single string, named list, dict), special-token
  injection, and the string-or-parts multimodal `content`.
- Emits a **prompt string**. Turning it into token IDs stays the caller's job (`tokenizers`,
  `tiktoken-rs`, …) — that boundary is deliberate.

## Verified compatibility

Every model below renders **byte-identical** to `transformers` in CI. See
[`COMPATIBILITY.md`](COMPATIBILITY.md) and the [corpus](tests/corpus/).

| Model | Notes |
|---|---|
| Qwen2.5, Qwen3 | ChatML, tool calling (`tojson`) |
| SmolLM2 | ChatML |
| Phi-3 | `<|user|>` / `<|end|>` markers |
| Hermes-3-Llama-3.1 | named `tool_use` sub-template, Jinja macros + recursion |

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `pycompat` | ✅ | Python methods on values (`.strip()`, `.split()`, `\| items`, …) via `minijinja-contrib`. Disable to drop that dependency if your templates don't need them. |

## Caveats

- **No automatic BOS doubling.** If a template emits `{{ bos_token }}`, set
  `add_special_tokens = false` at encode time so the tokenizer doesn't add BOS again. This crate
  renders exactly what the template says and never strips silently.
- **`strftime_now` defaults to UTC.** `transformers` uses local time. Inject a `FixedClock` (or
  your own `Clock`) when you need to match a specific reference exactly.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

The files under `tests/corpus/` are trimmed excerpts of upstream model configs, redistributed
under each model's own license — see `tests/corpus/README.md`.

[`RenderInput`]: https://docs.rs/hf-chat-template/latest/hf_chat_template/struct.RenderInput.html
