# hf-chat-template

[![CI](https://github.com/GregoryBolshakov/hf-chat-template/actions/workflows/ci.yml/badge.svg)](https://github.com/GregoryBolshakov/hf-chat-template/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/hf-chat-template.svg)](https://crates.io/crates/hf-chat-template)
[![docs.rs](https://img.shields.io/docsrs/hf-chat-template)](https://docs.rs/hf-chat-template)
[![license](https://img.shields.io/crates/l/hf-chat-template.svg)](#license)

Render a Hugging Face `chat_template` into a prompt string, byte-for-byte identical to Python's
`transformers.apply_chat_template`. The template is the Jinja2 string stored in a model's
`tokenizer_config.json`.

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

let prompt = tmpl.render_messages(&[Message::user("Hello!")], true)?;
assert_eq!(prompt, "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n");
# Ok::<(), hf_chat_template::Error>(())
```

Load a real model's config to inject its special tokens and resolve named templates:

```rust,no_run
use hf_chat_template::{ChatTemplate, Message, TokenizerConfig};

let json = std::fs::read_to_string("tokenizer_config.json")?;
let cfg: TokenizerConfig = serde_json::from_str(&json)?;
let tmpl = ChatTemplate::from_tokenizer_config(&cfg)?;

let prompt = tmpl.render_messages(&[Message::user("Hi")], true)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Newer models ship the template as a standalone `chat_template.jinja` file instead of inside
`tokenizer_config.json`. Load that with `from_template_and_config`, passing the template string
and the config the special tokens come from.

```rust,no_run
use hf_chat_template::{ChatTemplate, Message, TokenizerConfig};

let jinja = std::fs::read_to_string("chat_template.jinja")?;
let cfg: TokenizerConfig = serde_json::from_str(&std::fs::read_to_string("tokenizer_config.json")?)?;
let tmpl = ChatTemplate::from_template_and_config(&jinja, &cfg)?;

let prompt = tmpl.render_messages(&[Message::user("Hi")], true)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

For tools, documents, or model-specific kwargs, build a [`RenderInput`]. For an arbitrary
context, call `render_value` with a `minijinja::Value`.

## What it does

The Jinja engine is [`minijinja`](https://crates.io/crates/minijinja). This crate adds the
`transformers` compatibility layer on top of it, plus a corpus that checks byte-identical output
against real models on every commit.

It installs the globals that templates use: `raise_exception`, `strftime_now`, and a
Python-compatible `tojson` that matches the separators and key order of
`json.dumps(..., ensure_ascii=False)`. Python string, list, and dict methods come from `pycompat`.

It handles the three `chat_template` shapes (single string, named list, dict), special-token
injection, and the string-or-parts multimodal `content`.

It emits a prompt string. Turning that into token IDs stays the caller's job (`tokenizers`,
`tiktoken-rs`).

## Verified compatibility

These models render byte-identical to `transformers` in CI. See
[`COMPATIBILITY.md`](COMPATIBILITY.md) and the [corpus](tests/corpus/).

| Model | Notes |
|---|---|
| Qwen2.5, Qwen3, QwQ-32B | ChatML, tool calling (`tojson`), reasoning |
| SmolLM2 | ChatML |
| Phi-3 | `<\|user\|>` / `<\|end\|>` markers |
| Hermes-3-Llama-3.1 | named `tool_use` sub-template, Jinja macros and recursion |
| Mistral-7B-Instruct-v0.3 | `[INST]` / `[AVAILABLE_TOOLS]`, tool calling |
| DeepSeek-R1-Distill, deepseek-llm | reasoning (`<think>`), `User:` / `Assistant:` |
| OpenChat-3.5, Zephyr, Yi-1.5, Falcon | varied prompt formats, pycompat methods |
| LFM2 | standalone `chat_template.jinja` file, tool list (`tojson`) |
| SmolLM3 | standalone file, `{% generation %}` reasoning block |

Fifteen models, fifty cases, all byte-identical in CI.

## Loading from the Hub

The `hub` feature adds `from_hub`, which fetches a model's config and template and compiles it in
one call. It loads `tokenizer_config.json`, plus a standalone `chat_template.jinja` when the model
ships one (the standalone file wins over an inline `chat_template`, matching `transformers`). It
uses the synchronous `hf-hub` client with rustls, so there is no system OpenSSL dependency. Authentication follows `hf-hub`: the `HF_TOKEN` env var or the token
from `huggingface-cli login`, which gated repos need.

```toml
hf-chat-template = { version = "0.1", features = ["hub"] }
```

```rust,no_run
use hf_chat_template::{ChatTemplate, Message};

let tmpl = ChatTemplate::from_hub("Qwen/Qwen2.5-0.5B-Instruct")?;
let prompt = tmpl.render_messages(&[Message::user("Hi")], true)?;
# Ok::<(), hf_chat_template::Error>(())
```

## Tokenizing

The `tokenizers` feature adds `render_and_encode`, which renders the prompt and encodes it to
token IDs in one step. It encodes with `add_special_tokens = false`, because the template already
emits the model's special tokens. This is what `transformers.apply_chat_template(...,
tokenize=True)` does, and it avoids a doubled BOS.

```toml
hf-chat-template = { version = "0.1", features = ["tokenizers"] }
```

```rust,no_run
use hf_chat_template::{ChatTemplate, Message, RenderInput};
use hf_chat_template::tokenizers::Tokenizer;

let tmpl = ChatTemplate::from_str("{{ messages[0].content }}")?;
let tok = Tokenizer::from_file("tokenizer.json").unwrap();
let input = RenderInput { messages: vec![Message::user("hi")], ..Default::default() };
let (prompt, ids) = tmpl.render_and_encode(&input, &tok)?;
# Ok::<(), hf_chat_template::Error>(())
```

## Feature flags

`pycompat` is on by default. It adds Python methods on values (`.strip()`, `.split()`, `| items`)
through `minijinja-contrib`. Disable it to drop that dependency when your templates do not use
those methods.

`hub` is off by default. It adds `from_hub` and `from_hub_revision`, pulling in `hf-hub` and a
TLS stack that the core string-rendering path does not need.

`tokenizers` is off by default. It adds `render_and_encode` and re-exports `tokenizers`, pulling
in that crate and its `onig` regex backend.

## Caveats

This crate does not add or strip a BOS token. If a template emits `{{ bos_token }}`, set
`add_special_tokens = false` at encode time so the tokenizer does not add BOS a second time. It
renders what the template says.

`strftime_now` defaults to UTC, while `transformers` uses local time. Inject a `FixedClock` or
your own `Clock` to match a specific reference.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

Files under `tests/corpus/` are trimmed excerpts of upstream model configs, redistributed under
each model's own license. See `tests/corpus/README.md`.

[`RenderInput`]: https://docs.rs/hf-chat-template/latest/hf_chat_template/struct.RenderInput.html
