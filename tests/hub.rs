//! `hub` feature: fetch a real model config from the Hugging Face Hub and render it.
//!
//! Hits the network, so it is `#[ignore]`d by default and excluded from the published package.
//! Run it explicitly:
//!
//! ```text
//! cargo test --features hub --test hub -- --ignored
//! ```

#![cfg(feature = "hub")]

use hf_chat_template::{ChatTemplate, Message};

#[test]
#[ignore = "network: fetches tokenizer_config.json from the Hugging Face Hub"]
fn from_hub_renders_an_ungated_model() {
    let tmpl = ChatTemplate::from_hub("Qwen/Qwen2.5-0.5B-Instruct").expect("fetch + compile");
    let prompt = tmpl
        .render_messages(&[Message::user("Hi")], true)
        .expect("render");
    assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>"));
    assert!(prompt.ends_with("<|im_start|>assistant\n"));
}

#[test]
#[ignore = "network: fetches a pinned revision from the Hugging Face Hub"]
fn from_hub_revision_pins_a_commit() {
    let tmpl = ChatTemplate::from_hub_revision("Qwen/Qwen2.5-0.5B-Instruct", "main")
        .expect("fetch + compile");
    let prompt = tmpl
        .render_messages(&[Message::user("Hi")], true)
        .expect("render");
    assert!(prompt.contains("Hi"));
}
