# Golden corpus

Proof that `hf-chat-template` renders real Hugging Face chat templates **byte-for-byte**
identically to the Python reference, `transformers.apply_chat_template`. This is the crate's
core guarantee; `tests/m3_corpus.rs` fails CI on any divergence.

## Layout

```
<model-slug>/
  meta.json              model id, pinned revision (commit sha), license, source URL
  tokenizer_config.json  trimmed config (chat_template if inline, + special tokens)
  chat_template.jinja    optional: standalone template, for models that ship one (not inline)
  cases/<name>.json      { "template_name": null | "<name>", "input": <RenderInput> }
  expected/<name>.txt    byte-exact transformers output (committed; marked -text)
```

The Rust side renders each case via the public `ChatTemplate::builder_from_config` path, or via
`ChatTemplate::from_template_and_config` when a `chat_template.jinja` is present (the
standalone-file layout), and asserts equality with `expected/<name>.txt`. CI needs **no Python** —
only regenerating the references does.

## Reference provenance

Generated with **`transformers` 5.12.0** (Python 3.10, `jinja2` 3.1.6). Regenerate with
`tools/gen_reference.py`; re-pin the version here when you do.

| model | model_id | license | revision (sha) | cases |
|---|---|---|---|---|
| deepseek-llm-7b-chat | deepseek-ai/deepseek-llm-7b-chat | DeepSeek License Agreement | `afbda8b347ec` | 3 |
| deepseek-r1-distill-qwen-7b | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | MIT | `916b56a44061` | 3 |
| falcon-7b-instruct | tiiuae/falcon-7b-instruct | Apache-2.0 | `8782b5c5d8c9` | 3 |
| hermes-3-llama-3.1-8b | NousResearch/Hermes-3-Llama-3.1-8B | Llama-3.1-Community | `896ea440e5a9` | 4 |
| lfm2-1.2b | LiquidAI/LFM2-1.2B | LFM Open License v1.0 | `933cee00d754` | 4 |
| mistral-7b-instruct-v0.3 | mistralai/Mistral-7B-Instruct-v0.3 | Apache-2.0 | `c170c708c41d` | 4 |
| openchat-3.5-0106 | openchat/openchat-3.5-0106 | Apache-2.0 | `ff058fda4972` | 3 |
| phi-3-mini-4k-instruct | microsoft/Phi-3-mini-4k-instruct | MIT | `f39ac1d28e92` | 3 |
| qwen2.5-0.5b-instruct | Qwen/Qwen2.5-0.5B-Instruct | Apache-2.0 | `7ae557604adf` | 4 |
| qwen3-0.6b | Qwen/Qwen3-0.6B | Apache-2.0 | `c1899de289a0` | 4 |
| qwq-32b | Qwen/QwQ-32B | Apache-2.0 | `976055f8c83f` | 4 |
| smollm2-1.7b-instruct | HuggingFaceTB/SmolLM2-1.7B-Instruct | Apache-2.0 | `31b70e2e869a` | 3 |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B | Apache-2.0 | `a07cc9a04f16` | 2 |
| yi-1.5-9b-chat | 01-ai/Yi-1.5-9B-Chat | Apache-2.0 | `1a0fc698cf88` | 3 |
| zephyr-7b-beta | HuggingFaceH4/zephyr-7b-beta | MIT | `892b3d7a7b1c` | 3 |

Each `tokenizer_config.json` is a trimmed excerpt of the upstream model's file (`chat_template`
plus special tokens), redistributed under that model's license — see each row's `license` and
`meta.json` `source`. Only the template/token metadata is included; no model weights.

## Coverage notes

The generic cases (`basic`, `no_system`, `single_user`) are date-independent on purpose — no
model here uses `strftime_now`, so references are reproducible without clock pinning. The
`with_tools` / `tool_use` cases exercise `tojson` key-order and (for Hermes) the named `tool_use`
sub-template with Jinja macros. `lfm2-1.2b` covers the standalone `chat_template.jinja` layout,
where the template lives in its own file and the special tokens come from `tokenizer_config.json`.
`smollm3-3b` covers the `transformers` `{% generation %}` block: its cases use a `/system_override`
system message, which takes the template's override branch (no `strftime_now`, so date-independent)
while assistant messages still exercise the generation block.

The 2026-06 breadth pass added the major ungated template families: Mistral `[INST]` /
`[AVAILABLE_TOOLS]` (`mistral-7b-instruct-v0.3`), reasoning templates that split on `</think>`
(`qwq-32b`, `deepseek-r1-distill-qwen-7b`), DeepSeek `User:` / `Assistant:` (`deepseek-llm-7b-chat`),
OpenChat's `.title()`-cased roles (`openchat-3.5-0106`), Zephyr `<|user|>` markers (`zephyr-7b-beta`),
a ChatML variant (`yi-1.5-9b-chat`), and Falcon's `.strip()` / `.replace()` (`falcon-7b-instruct`).
All are date-independent. Models that use `strftime_now` (Granite, Llama-3.1, Command-R) and the
gated families (Llama-3.x, Gemma, Command-R) are still pending: the former on the `strftime`
real-clock feature, the latter on an `HF_TOKEN` to fetch.
