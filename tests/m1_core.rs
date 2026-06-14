//! M1 acceptance tests: the core render path + compat shims, against real-world template
//! shapes (ChatML, Mistral) and each shim's contract.

use hf_chat_template::{ChatTemplate, Error, FixedClock};
use minijinja::{context, Value};

/// Helper: a list of {role, content} messages as a minijinja Value.
fn messages(pairs: &[(&str, &str)]) -> Value {
    let v: Vec<Value> = pairs
        .iter()
        .map(|(r, c)| context! { role => *r, content => *c })
        .collect();
    Value::from(v)
}

/// ChatML (Qwen/many OSS models): `<|im_start|>role\ncontent<|im_end|>\n`, with a generation
/// prompt opening the assistant turn.
#[test]
fn chatml_basic_with_generation_prompt() {
    let src = "{% for message in messages %}\
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}\
{% endfor %}\
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    let tmpl = ChatTemplate::from_str(src).unwrap();
    let out = tmpl
        .render_value(context! {
            messages => messages(&[
                ("system", "You are helpful."),
                ("user", "Hello"),
            ]),
            add_generation_prompt => true,
        })
        .unwrap();

    let expected = "<|im_start|>system\nYou are helpful.<|im_end|>\n\
<|im_start|>user\nHello<|im_end|>\n\
<|im_start|>assistant\n";
    assert_eq!(out, expected);
}

/// Mistral-style `[INST] ... [/INST]` with bos_token from context and eos after each reply.
/// Exercises the alternating-role loop and special-token context variables.
#[test]
fn mistral_inst_with_bos_eos() {
    let src = "{{ bos_token }}\
{% for message in messages %}\
{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}\
{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}\
{% endif %}\
{% endfor %}";

    let tmpl = ChatTemplate::from_str(src).unwrap();
    let out = tmpl
        .render_value(context! {
            messages => messages(&[
                ("user", "Hi"),
                ("assistant", "Hello!"),
                ("user", "Bye"),
            ]),
            bos_token => "<s>",
            eos_token => "</s>",
        })
        .unwrap();

    assert_eq!(out, "<s>[INST] Hi [/INST] Hello!</s>[INST] Bye [/INST]");
}

/// The tojson divergence M0 found: must preserve insertion order and use Python's `", "` /
/// `": "` separators — NOT minijinja's sorted/space-less builtin.
#[test]
fn tojson_matches_python_separators_and_order() {
    let tmpl = ChatTemplate::from_str("{{ tool | tojson }}").unwrap();
    // Realistic path: tool data arrives as JSON, deserialized straight into a minijinja Value,
    // which (with preserve_order) keeps insertion order. (`context!` would sort keys.)
    let tool: Value = serde_json::from_str(
        r#"{"name":"get_weather","arguments":{"city":"Paris","units":"celsius"}}"#,
    )
    .unwrap();
    let out = tmpl.render_value(context! { tool => tool }).unwrap();
    // Python json.dumps(..., ensure_ascii=False) style: spaces after : and , ; insertion order.
    assert_eq!(
        out,
        r#"{"name": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}"#
    );
}

/// tojson with indent renders pretty, Python-style.
#[test]
fn tojson_with_indent() {
    let tmpl = ChatTemplate::from_str("{{ obj | tojson(indent=2) }}").unwrap();
    let out = tmpl
        .render_value(context! { obj => context!{ a => 1, b => 2 } })
        .unwrap();
    assert_eq!(out, "{\n  \"a\": 1,\n  \"b\": 2\n}");
}

/// raise_exception must surface as the distinct `TemplateRaised` variant carrying the message,
/// never as a generic render error — callers branch on this.
#[test]
fn raise_exception_becomes_template_raised() {
    let src = "{% for m in messages %}\
{% if m['role'] not in ['user','assistant'] %}\
{{ raise_exception('Unknown role: ' + m['role']) }}\
{% endif %}{{ m['content'] }}{% endfor %}";

    let tmpl = ChatTemplate::from_str(src).unwrap();
    let err = tmpl
        .render_value(context! { messages => messages(&[("system", "x")]) })
        .unwrap_err();

    match err {
        Error::TemplateRaised { message } => {
            assert_eq!(message, "Unknown role: system");
        }
        other => panic!("expected TemplateRaised, got {other:?}"),
    }
}

/// strftime_now uses the injected clock — pinned date gives byte-stable output.
#[test]
fn strftime_now_uses_fixed_clock() {
    let tmpl = ChatTemplate::builder("Today: {{ strftime_now('%d %B %Y') }}")
        .clock(FixedClock::from_ymd(2026, 6, 13).unwrap())
        .build()
        .unwrap();
    let out = tmpl.render_value(context! {}).unwrap();
    assert_eq!(out, "Today: 13 June 2026");
}

/// pycompat: Python string methods used in real templates work.
#[test]
fn pycompat_string_methods_in_template() {
    let tmpl =
        ChatTemplate::from_str("{{ content.strip().title() }}|{{ role.startswith('assi') }}")
            .unwrap();
    let out = tmpl
        .render_value(context! { content => "  hello world  ", role => "assistant" })
        .unwrap();
    assert_eq!(out, "Hello World|true");
}

/// A syntactically broken template fails at construction with Compile, not at render.
#[test]
fn syntax_error_is_compile_error() {
    let err = ChatTemplate::from_str("{% for x in %}").unwrap_err();
    assert!(matches!(err, Error::Compile(_)), "got {err:?}");
}

/// FromStr parity with from_str.
#[test]
fn fromstr_trait_works() {
    let tmpl: ChatTemplate = "hi {{ name }}".parse().unwrap();
    assert_eq!(
        tmpl.render_value(context! { name => "bob" }).unwrap(),
        "hi bob"
    );
}
