//! `tokenizers` feature: render then encode, and the no-double-special-tokens contract.
//!
//! Offline: uses a tiny hand-built WordLevel tokenizer (`tests/fixtures/tiny_tokenizer.json`)
//! whose post-processor prepends `[BOS]` (id 3) only when `add_special_tokens = true`.

#![cfg(feature = "tokenizers")]

use hf_chat_template::tokenizers::Tokenizer;
use hf_chat_template::{ChatTemplate, Message, RenderInput};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tiny_tokenizer.json"
);

#[test]
fn render_and_encode_returns_prompt_and_ids() {
    let tok = Tokenizer::from_file(FIXTURE).expect("load fixture tokenizer");
    let tmpl = ChatTemplate::from_str("{{ messages[0].content }}").unwrap();
    let input = RenderInput {
        messages: vec![Message::user("hello world")],
        ..Default::default()
    };

    let (prompt, ids) = tmpl.render_and_encode(&input, &tok).unwrap();
    assert_eq!(prompt, "hello world");
    assert_eq!(ids, vec![0, 1]);
}

#[test]
fn render_and_encode_does_not_double_special_tokens() {
    let tok = Tokenizer::from_file(FIXTURE).expect("load fixture tokenizer");
    let tmpl = ChatTemplate::from_str("{{ messages[0].content }}").unwrap();
    let input = RenderInput {
        messages: vec![Message::user("hello world")],
        ..Default::default()
    };

    let (_, ids) = tmpl.render_and_encode(&input, &tok).unwrap();
    // The template owns the special tokens, so we encode with add_special_tokens = false:
    // [BOS] (id 3) is not prepended.
    assert_eq!(ids, vec![0, 1]);

    // Contrast: the same tokenizer with add_special_tokens = true would prepend [BOS].
    let with_specials = tok.encode("hello world", true).unwrap();
    assert_eq!(with_specials.get_ids(), &[3, 0, 1]);
}
