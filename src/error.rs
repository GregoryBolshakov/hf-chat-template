//! Error model. One enum; template-raised rejections are distinguished from engine bugs so
//! callers can act on them (reject the conversation) vs. log a library issue.

use std::fmt;

/// Private marker prefixed onto messages passed to `raise_exception` so we can reliably
/// recover them from minijinja's error chain without shared mutable state. Never appears in
/// successful render output (a raise always produces an error, not text).
pub(crate) const RAISE_SENTINEL: &str = "\u{0}HFCT_RAISE\u{1}";

/// Errors produced while compiling or rendering a chat template.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The template deliberately called `raise_exception(msg)` — the input conversation was
    /// rejected by the template's own validation (e.g. bad role alternation). Act on this;
    /// it is not a bug.
    TemplateRaised {
        /// The message the template passed to `raise_exception`.
        message: String,
    },
    /// The Jinja source failed to compile (syntax error). Carries the underlying engine error.
    Compile(minijinja::Error),
    /// The template failed at render time (undefined variable, type error, unknown method…).
    Render(minijinja::Error),
    /// The `tokenizer_config.json` had no usable `chat_template`, or a requested named
    /// template was absent / the field had an unexpected shape.
    Config(String),
    /// Fetching a file from the Hugging Face Hub failed (network, auth/gated repo, or the file
    /// is absent). Only present with the `hub` feature. Carries the underlying message; the
    /// `hf-hub` error type is deliberately kept out of our public API.
    #[cfg(feature = "hub")]
    Hub(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::TemplateRaised { message } => {
                write!(f, "template rejected the input: {message}")
            }
            Error::Compile(e) => write!(f, "chat template failed to compile: {e}"),
            Error::Render(e) => write!(f, "chat template failed to render: {e}"),
            Error::Config(m) => write!(f, "invalid chat-template config: {m}"),
            #[cfg(feature = "hub")]
            Error::Hub(m) => write!(f, "failed to fetch from the Hugging Face Hub: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Compile(e) | Error::Render(e) => Some(e),
            _ => None,
        }
    }
}

impl Error {
    /// Map a minijinja render-time error into our error model, recovering a template-raised
    /// message (marked with [`RAISE_SENTINEL`]) into [`Error::TemplateRaised`].
    pub(crate) fn from_render(e: minijinja::Error) -> Self {
        // Walk the message and the error source chain looking for our sentinel.
        if let Some(msg) = find_raised_message(&e) {
            return Error::TemplateRaised { message: msg };
        }
        Error::Render(e)
    }
}

/// Scan a minijinja error (and its source chain) for a sentinel-marked raise message.
fn find_raised_message(e: &minijinja::Error) -> Option<String> {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(err) = current {
        let s = err.to_string();
        if let Some(idx) = s.find(RAISE_SENTINEL) {
            let after = &s[idx + RAISE_SENTINEL.len()..];
            // The raised message runs to the next sentinel (if minijinja appended context)
            // or to end of string.
            let msg = match after.find(RAISE_SENTINEL) {
                Some(end) => &after[..end],
                None => after,
            };
            return Some(msg.to_string());
        }
        current = err.source();
    }
    None
}
