//! The public [`ChatTemplate`] type and its builder.

use std::sync::Arc;

use minijinja::{Environment, UndefinedBehavior, Value};
use serde_json::{Map, Value as Json};

use crate::clock::{Clock, SystemClock};
use crate::config::{self, ChatTemplateField, TokenizerConfig};
use crate::engine::{self, EngineConfig};
use crate::error::Error;
use crate::model::{Message, RenderInput};

/// A compiled Hugging Face chat template, ready to render.
///
/// Construction compiles the Jinja source and installs the `transformers`-compatible
/// globals/filters. [`render`](ChatTemplate::render) borrows `&self`, so a single instance is
/// cheap to reuse and shareable across threads.
pub struct ChatTemplate {
    env: Environment<'static>,
    /// Special tokens captured from a [`TokenizerConfig`] (empty for raw-string construction),
    /// injected into the render context as `{{ bos_token }}` etc. unless the input overrides them.
    special_tokens: Vec<(String, String)>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The compiled environment is large and not user-meaningful; keep this opaque.
        f.debug_struct("ChatTemplate")
            .field("special_tokens", &self.special_tokens)
            .finish_non_exhaustive()
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

    /// Compile from a parsed `tokenizer_config.json`, resolving the default template and
    /// injecting the config's special tokens (`bos_token`, …) into the render context.
    ///
    /// For named-template selection (`tool_use`, `rag`) or custom options, use
    /// [`builder_from_config`](ChatTemplate::builder_from_config).
    pub fn from_tokenizer_config(config: &TokenizerConfig) -> Result<Self, Error> {
        ChatTemplate::builder_from_config(config)?.build()
    }

    /// Start a builder to compile `source` with non-default options (clock, undefined policy…).
    pub fn builder(source: &str) -> ChatTemplateBuilder {
        ChatTemplateBuilder::new(BuilderSource::Raw(source.to_owned()), Vec::new())
    }

    /// Start a builder from a `tokenizer_config.json`, carrying its special tokens. Use
    /// [`template_name`](ChatTemplateBuilder::template_name) to pick a named sub-template.
    ///
    /// Fails with [`Error::Config`] if the config has no `chat_template` field at all.
    pub fn builder_from_config(config: &TokenizerConfig) -> Result<ChatTemplateBuilder, Error> {
        let field = config
            .chat_template
            .clone()
            .ok_or_else(|| Error::Config("tokenizer config has no chat_template".into()))?;
        Ok(ChatTemplateBuilder::new(
            BuilderSource::Field(field),
            config.special_tokens(),
        ))
    }

    /// Fetch `tokenizer_config.json` for a Hub repo (default branch) and compile its default
    /// template, injecting the config's special tokens. Requires the `hub` feature.
    ///
    /// Authentication uses `hf-hub`'s discovery (the `HF_TOKEN` env var or the cached
    /// `huggingface-cli login` token); gated repos need a token with access. For a pinned
    /// commit or branch, use [`from_hub_revision`](ChatTemplate::from_hub_revision).
    ///
    /// ```no_run
    /// use hf_chat_template::{ChatTemplate, Message};
    /// let tmpl = ChatTemplate::from_hub("Qwen/Qwen2.5-0.5B-Instruct")?;
    /// let prompt = tmpl.render_messages(&[Message::user("Hi")], true)?;
    /// # Ok::<(), hf_chat_template::Error>(())
    /// ```
    #[cfg(feature = "hub")]
    pub fn from_hub(repo_id: &str) -> Result<Self, Error> {
        let config = crate::hub::fetch_config(repo_id, None)?;
        ChatTemplate::from_tokenizer_config(&config)
    }

    /// Like [`from_hub`](ChatTemplate::from_hub), but pins a specific `revision` (a branch,
    /// tag, or commit SHA). Requires the `hub` feature.
    #[cfg(feature = "hub")]
    pub fn from_hub_revision(repo_id: &str, revision: &str) -> Result<Self, Error> {
        let config = crate::hub::fetch_config(repo_id, Some(revision))?;
        ChatTemplate::from_tokenizer_config(&config)
    }

    /// Render the typed input model to the final prompt string. Special tokens from the
    /// source config are injected first; any matching key in [`RenderInput::extra`] wins.
    pub fn render(&self, input: &RenderInput) -> Result<String, Error> {
        let ctx = self.build_context(input)?;
        self.render_value(ctx)
    }

    /// Render `input`, then encode the prompt to token IDs with `tokenizer`, returning both.
    /// Requires the `tokenizers` feature.
    ///
    /// Encodes with `add_special_tokens = false`: a chat template already emits the model's
    /// special tokens (`bos_token`, end-of-turn markers, …), so letting the tokenizer add them
    /// again would double them. This matches what `transformers.apply_chat_template(...,
    /// tokenize=True)` does. The `tokenizer` you pass is the model's own `tokenizer.json`
    /// (e.g. `tokenizers::Tokenizer::from_file("tokenizer.json")`).
    ///
    /// ```no_run
    /// use hf_chat_template::{ChatTemplate, Message, RenderInput};
    /// use hf_chat_template::tokenizers::Tokenizer;
    ///
    /// let tmpl = ChatTemplate::from_str("{{ messages[0].content }}")?;
    /// let tok = Tokenizer::from_file("tokenizer.json").unwrap();
    /// let input = RenderInput { messages: vec![Message::user("hi")], ..Default::default() };
    /// let (prompt, ids) = tmpl.render_and_encode(&input, &tok)?;
    /// # Ok::<(), hf_chat_template::Error>(())
    /// ```
    #[cfg(feature = "tokenizers")]
    pub fn render_and_encode(
        &self,
        input: &RenderInput,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Result<(String, Vec<u32>), Error> {
        let prompt = self.render(input)?;
        let encoding = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| Error::Tokenize(e.to_string()))?;
        let ids = encoding.get_ids().to_vec();
        Ok((prompt, ids))
    }

    /// Convenience: render with only `messages` and the generation-prompt flag.
    pub fn render_messages(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String, Error> {
        let input = RenderInput {
            messages: messages.to_vec(),
            add_generation_prompt,
            ..Default::default()
        };
        self.render(&input)
    }

    /// Render with an arbitrary minijinja context value — the low-level escape hatch for
    /// callers who need to pass template variables we don't model with typed structs.
    ///
    /// Unlike [`render`](ChatTemplate::render), this does **not** inject the config's special
    /// tokens; the caller's context is used as-is.
    pub fn render_value(&self, ctx: Value) -> Result<String, Error> {
        let tmpl = self
            .env
            .get_template("chat")
            .expect("the 'chat' template is always present after construction");
        tmpl.render(ctx).map_err(Error::from_render)
    }

    /// Build the Jinja context: special tokens, then the serialized input on top (input wins).
    /// Routed through `serde_json` to a single `minijinja::Value`; `preserve_order` on both
    /// crates keeps map key order intact end-to-end (load-bearing for `| tojson`).
    fn build_context(&self, input: &RenderInput) -> Result<Value, Error> {
        let mut ctx: Map<String, Json> = Map::new();
        for (k, v) in &self.special_tokens {
            ctx.insert(k.clone(), Json::String(v.clone()));
        }
        let serialized = serde_json::to_value(input)
            .map_err(|e| Error::Config(format!("failed to serialize render input: {e}")))?;
        if let Json::Object(map) = serialized {
            for (k, v) in map {
                ctx.insert(k, v);
            }
        }
        Ok(Value::from_serialize(Json::Object(ctx)))
    }
}

/// Where a builder draws its template source from.
enum BuilderSource {
    /// A raw Jinja string.
    Raw(String),
    /// A `chat_template` field whose concrete source is resolved at `build()`.
    Field(ChatTemplateField),
}

/// Builder for [`ChatTemplate`], exposing the knobs that affect rendering semantics.
///
/// Defaults mirror the `transformers` reference environment:
/// `trim_blocks = true`, `lstrip_blocks = true`, `keep_trailing_newline = true`,
/// lenient undefined behavior, pycompat enabled, [`SystemClock`].
pub struct ChatTemplateBuilder {
    source: BuilderSource,
    special_tokens: Vec<(String, String)>,
    template_name: Option<String>,
    cfg: EngineConfig,
}

impl ChatTemplateBuilder {
    fn new(source: BuilderSource, special_tokens: Vec<(String, String)>) -> Self {
        ChatTemplateBuilder {
            source,
            special_tokens,
            template_name: None,
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

    /// Select a named sub-template (e.g. `"tool_use"`, `"rag"`) when the config carries a
    /// list of them. Ignored for a single-string source.
    pub fn template_name(mut self, name: impl Into<String>) -> Self {
        self.template_name = Some(name.into());
        self
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

    /// Compile the template. Returns [`Error::Compile`] on a Jinja syntax error, or
    /// [`Error::Config`] if a requested/needed named template can't be resolved.
    pub fn build(self) -> Result<ChatTemplate, Error> {
        let source: String = match &self.source {
            BuilderSource::Raw(s) => s.clone(),
            BuilderSource::Field(field) => {
                config::resolve_template(field, self.template_name.as_deref())?.to_owned()
            }
        };
        let env = engine::build(source, &self.cfg).map_err(Error::Compile)?;
        Ok(ChatTemplate {
            env,
            special_tokens: self.special_tokens,
        })
    }
}

impl std::str::FromStr for ChatTemplate {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        ChatTemplate::from_str(s)
    }
}
