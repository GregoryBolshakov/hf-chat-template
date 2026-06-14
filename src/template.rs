//! The public [`ChatTemplate`] type and its builder.

use std::sync::Arc;

use minijinja::{Environment, UndefinedBehavior, Value};

use crate::clock::{Clock, SystemClock};
use crate::engine::{self, EngineConfig};
use crate::error::Error;

/// A compiled Hugging Face chat template, ready to render.
///
/// Construction compiles the Jinja source and installs the `transformers`-compatible
/// globals/filters. [`render_value`](ChatTemplate::render_value) borrows `&self`, so a single
/// instance is cheap to reuse and shareable across threads.
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The compiled environment is large and not user-meaningful; keep this opaque.
        f.debug_struct("ChatTemplate").finish_non_exhaustive()
    }
}

impl ChatTemplate {
    /// Compile from a raw Jinja chat-template string, using the default
    /// `transformers`-compatible settings (see [`ChatTemplateBuilder`] to customize).
    ///
    /// An inherent `from_str` (not just the [`FromStr`](std::str::FromStr) impl) so callers
    /// can write `ChatTemplate::from_str(s)` without importing the trait.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(source: &str) -> Result<Self, Error> {
        ChatTemplate::builder(source).build()
    }

    /// Start a builder to compile `source` with non-default options (clock, undefined policy…).
    pub fn builder(source: &str) -> ChatTemplateBuilder {
        ChatTemplateBuilder::new(source)
    }

    /// Render with an arbitrary minijinja context value — the low-level escape hatch for
    /// callers who need to pass template variables we don't model with typed structs.
    ///
    /// The context is typically a map of `messages`, `tools`, `add_generation_prompt`, plus
    /// any special-token variables (`bos_token`, …) and model-specific extras.
    pub fn render_value(&self, ctx: Value) -> Result<String, Error> {
        let tmpl = self
            .env
            .get_template("chat")
            .expect("the 'chat' template is always present after construction");
        tmpl.render(ctx).map_err(Error::from_render)
    }
}

/// Builder for [`ChatTemplate`], exposing the knobs that affect rendering semantics.
///
/// Defaults mirror the `transformers` reference environment:
/// `trim_blocks = true`, `lstrip_blocks = true`, `keep_trailing_newline = true`,
/// lenient undefined behavior, pycompat enabled, [`SystemClock`].
pub struct ChatTemplateBuilder {
    source: String,
    cfg: EngineConfig,
}

impl ChatTemplateBuilder {
    fn new(source: &str) -> Self {
        ChatTemplateBuilder {
            source: source.to_owned(),
            cfg: EngineConfig {
                // VERIFY against transformers source; pinned here as the documented baseline.
                trim_blocks: true,
                lstrip_blocks: true,
                keep_trailing_newline: true,
                undefined: UndefinedBehavior::Lenient,
                pycompat: true,
                clock: Arc::new(SystemClock),
            },
        }
    }

    /// Inject a deterministic [`Clock`] for `strftime_now` (essential for golden tests).
    pub fn clock(mut self, clock: impl Clock + 'static) -> Self {
        self.cfg.clock = Arc::new(clock);
        self
    }

    /// Enable/disable the pycompat Python-method shim. Default: enabled.
    pub fn pycompat(mut self, enabled: bool) -> Self {
        self.cfg.pycompat = enabled;
        self
    }

    /// Override how undefined variables are treated. Default: [`UndefinedBehavior::Lenient`].
    pub fn undefined_behavior(mut self, ub: UndefinedBehavior) -> Self {
        self.cfg.undefined = ub;
        self
    }

    /// Compile the template. Returns [`Error::Compile`] on a Jinja syntax error.
    pub fn build(self) -> Result<ChatTemplate, Error> {
        let env = engine::build(self.source, &self.cfg).map_err(Error::Compile)?;
        Ok(ChatTemplate { env })
    }
}

impl std::str::FromStr for ChatTemplate {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        ChatTemplate::from_str(s)
    }
}
