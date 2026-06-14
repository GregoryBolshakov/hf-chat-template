//! Builds a `minijinja::Environment` configured to match the `transformers` chat-template
//! environment: HF globals, the Python-compatible `tojson`, pycompat method shim, the
//! `strftime_now` clock hook, and the whitespace-control flags.

use std::sync::Arc;

use minijinja::{AutoEscape, Environment, Error, ErrorKind, UndefinedBehavior, Value};

use crate::clock::Clock;
use crate::error::RAISE_SENTINEL;
use crate::json::tojson_filter;

/// Knobs that affect rendering semantics. Defaults mirror the `transformers` reference.
#[derive(Clone)]
pub(crate) struct EngineConfig {
    pub trim_blocks: bool,
    pub lstrip_blocks: bool,
    pub keep_trailing_newline: bool,
    pub undefined: UndefinedBehavior,
    pub pycompat: bool,
    pub clock: Arc<dyn Clock>,
}

/// Install all HF-compat globals, filters and settings onto a fresh environment, then add the
/// template source under the fixed name `"chat"`.
///
/// `'static` source: we own the template string for the lifetime of the environment, so we
/// build an `Environment<'static>` by handing it an owned `String`.
pub(crate) fn build(source: String, cfg: &EngineConfig) -> Result<Environment<'static>, Error> {
    let mut env = Environment::new();

    env.set_trim_blocks(cfg.trim_blocks);
    env.set_lstrip_blocks(cfg.lstrip_blocks);
    env.set_keep_trailing_newline(cfg.keep_trailing_newline);
    env.set_undefined_behavior(cfg.undefined);

    // Chat templates are not HTML; escaping would corrupt the prompt. Force no auto-escape.
    env.set_auto_escape_callback(|_name| AutoEscape::None);

    // Python string/list/dict methods (.strip(), .split(), .items(), ...). The shim lives in
    // minijinja-contrib, which is an optional dependency gated by the `pycompat` feature.
    let _pycompat = cfg.pycompat;
    #[cfg(feature = "pycompat")]
    if _pycompat {
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    }

    // raise_exception(msg): abort the render carrying `msg`, marked so we can recover it as a
    // distinct TemplateRaised error (not an engine bug). The sentinel never reaches output.
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        Err(Error::new(
            ErrorKind::InvalidOperation,
            format!("{RAISE_SENTINEL}{msg}{RAISE_SENTINEL}"),
        ))
    });

    // strftime_now(fmt): current date/time formatted via the injected clock.
    let clock = cfg.clock.clone();
    env.add_function("strftime_now", move |fmt: String| -> Result<Value, Error> {
        Ok(Value::from(clock.strftime(&fmt)))
    });

    // Python-compatible tojson (overrides minijinja's sorted/space-less builtin).
    env.add_filter("tojson", tojson_filter);

    env.add_template_owned("chat", source)?;

    Ok(env)
}
