//! M3 — golden corpus runner.
//!
//! For every `tests/corpus/<model>/`:
//!   * load the real (trimmed) `tokenizer_config.json`,
//!   * for each `cases/<name>.json` (a `{template_name, input}` wrapper), render with
//!     `hf-chat-template`, and
//!   * assert **byte-for-byte** equality against `expected/<name>.txt`, the output of
//!     `transformers.apply_chat_template` (see `tools/gen_reference.py`).
//!
//! This is the crate's moat: proof we match the Python reference on real model templates.
//! CI runs this without any Python — only regenerating the references needs `transformers`.

use std::fs;
use std::path::{Path, PathBuf};

use hf_chat_template::{
    ChatTemplate, ChatTemplateBuilder, Error, Message, RenderInput, TokenizerConfig,
};
use serde::Deserialize;

fn corpus_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus")
}

/// Build a [`ChatTemplateBuilder`] for a model dir, supporting both template layouts: an inline
/// `chat_template` in `tokenizer_config.json`, or a standalone `chat_template.jinja` file
/// alongside it (newer models — LFM2, Gemma 3+). The standalone file, when present, supplies the
/// body while special tokens still come from the config (matching `transformers`).
fn builder_for(dir: &Path, cfg: &TokenizerConfig) -> Result<ChatTemplateBuilder, Error> {
    let jinja = dir.join("chat_template.jinja");
    if jinja.exists() {
        let src =
            fs::read_to_string(&jinja).unwrap_or_else(|e| panic!("read {}: {e}", jinja.display()));
        Ok(ChatTemplate::builder(&src).special_tokens_from(cfg))
    } else {
        ChatTemplate::builder_from_config(cfg)
    }
}

/// One corpus case: the typed render input plus an optional named-template selection.
#[derive(Deserialize)]
struct Case {
    #[serde(default)]
    template_name: Option<String>,
    input: RenderInput,
}

fn model_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(corpus_dir())
        .expect("corpus dir exists")
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs
}

fn load_config(dir: &Path) -> TokenizerConfig {
    let path = dir.join("tokenizer_config.json");
    let raw = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

#[test]
fn corpus_matches_transformers_reference() {
    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for dir in model_dirs() {
        let slug = dir.file_name().unwrap().to_string_lossy().into_owned();
        let cfg = load_config(&dir);

        let cases_dir = dir.join("cases");
        let mut case_files: Vec<PathBuf> = fs::read_dir(&cases_dir)
            .unwrap_or_else(|e| panic!("read {}: {e}", cases_dir.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        case_files.sort();

        for cf in case_files {
            let name = cf.file_stem().unwrap().to_string_lossy().into_owned();
            let case: Case = serde_json::from_str(&fs::read_to_string(&cf).unwrap())
                .unwrap_or_else(|e| panic!("parse case {}: {e}", cf.display()));

            let expected_path = dir.join("expected").join(format!("{name}.txt"));
            let expected = match fs::read_to_string(&expected_path) {
                Ok(s) => s,
                Err(_) => {
                    failures.push(format!(
                        "{slug}/{name}: missing reference {} — run tools/gen_reference.py",
                        expected_path.display()
                    ));
                    continue;
                }
            };

            // Build through the real config path; honor a named sub-template if the case asks.
            let mut builder = match builder_for(&dir, &cfg) {
                Ok(b) => b,
                Err(e) => {
                    failures.push(format!("{slug}/{name}: builder_for failed: {e}"));
                    continue;
                }
            };
            if let Some(tn) = &case.template_name {
                builder = builder.template_name(tn);
            }
            let tmpl = match builder.build() {
                Ok(t) => t,
                Err(e) => {
                    failures.push(format!("{slug}/{name}: compile failed: {e}"));
                    continue;
                }
            };

            match tmpl.render(&case.input) {
                Ok(got) => {
                    checked += 1;
                    if got != expected {
                        failures.push(format!(
                            "{slug}/{name}: MISMATCH\n{}",
                            first_diff(&expected, &got)
                        ));
                    }
                }
                Err(e) => failures.push(format!("{slug}/{name}: render failed: {e}")),
            }
        }
    }

    assert!(checked > 0, "no corpus cases were checked");
    assert!(
        failures.is_empty(),
        "corpus byte-equality failures ({} checked):\n\n{}",
        checked,
        failures.join("\n\n")
    );
    eprintln!("corpus: {checked} cases byte-identical to transformers reference");
}

/// Smoke check kept separate: every real template must at least compile and render non-empty
/// against a generic conversation, even before/without a committed reference.
#[test]
fn every_real_template_compiles_and_renders() {
    let basic = RenderInput {
        messages: vec![
            Message::new("system", "You are terse."),
            Message::new("user", "Hello!"),
            Message::new("assistant", "Hi."),
            Message::new("user", "What is 2+2?"),
        ],
        add_generation_prompt: true,
        ..Default::default()
    };
    let mut failures = Vec::new();
    for dir in model_dirs() {
        let slug = dir.file_name().unwrap().to_string_lossy().into_owned();
        let cfg = load_config(&dir);
        match builder_for(&dir, &cfg)
            .and_then(|b| b.build())
            .and_then(|t| t.render(&basic))
        {
            Ok(out) if !out.is_empty() => {}
            Ok(_) => failures.push(format!("{slug}: empty output")),
            Err(e) => failures.push(format!("{slug}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "smoke failures:\n{}",
        failures.join("\n")
    );
}

/// Render the first byte-level divergence between `expected` and `got`, with whitespace made
/// visible — whitespace bugs dominate chat-template mismatches.
fn first_diff(expected: &str, got: &str) -> String {
    let eb = expected.as_bytes();
    let gb = got.as_bytes();
    let at = eb
        .iter()
        .zip(gb)
        .position(|(a, b)| a != b)
        .unwrap_or(eb.len().min(gb.len()));
    let lo = at.saturating_sub(40);
    let show = |s: &str, end: usize| -> String {
        let start = lo.min(s.len());
        let stop = end.min(s.len());
        escape_ws(&s[start..stop])
    };
    format!(
        "  first diff at byte {at} (expected {} bytes, got {} bytes)\n  expected: …{}\n  got:      …{}",
        eb.len(),
        gb.len(),
        show(expected, at + 40),
        show(got, at + 40),
    )
}

fn escape_ws(s: &str) -> String {
    s.replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
}
