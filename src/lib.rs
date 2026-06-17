//! Render Hugging Face `chat_template` (Jinja2) strings the way Python
//! `transformers.apply_chat_template` does.
//!
//! This crate emits a **prompt string**. Turning it into token IDs is the caller's job
//! (`tokenizers`, `tiktoken-rs`, …) — keeping that boundary is deliberate.
//!
//! ```
//! use hf_chat_template::ChatTemplate;
//! use minijinja::context;
//!
//! let tmpl = ChatTemplate::from_str(
//!     "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}\
//!      {% if add_generation_prompt %}<|assistant|>{% endif %}",
//! ).unwrap();
//!
//! let out = tmpl.render_value(context! {
//!     messages => vec![
//!         context!{ role => "user", content => "hi" },
//!     ],
//!     add_generation_prompt => true,
//! }).unwrap();
//! assert_eq!(out, "<|user|>hi\n<|assistant|>");
//! ```
//!
//! ## Special tokens & BOS doubling
//! Templates that emit `{{ bos_token }}` expect you to pass `bos_token` in the context, and
//! to set `add_special_tokens = false` at encode time so the tokenizer does not add BOS a
//! second time. This crate renders exactly what the template says and never strips silently.

#![forbid(unsafe_code)]

mod clock;
mod config;
mod engine;
mod error;
#[cfg(feature = "hub")]
mod hub;
mod json;
mod model;
mod template;

pub use clock::{Clock, FixedClock, SystemClock};
pub use config::{ChatTemplateField, NamedTemplate, TokenField, TokenizerConfig};
pub use error::Error;
pub use model::{Content, Message, RenderInput};
pub use template::{ChatTemplate, ChatTemplateBuilder};

/// Re-export of the `minijinja` we build against, so downstreams can construct context
/// [`Value`](minijinja::Value)s without guessing a version. Note: `render_value` ties our
/// public API to this `minijinja` major version.
pub use minijinja;
