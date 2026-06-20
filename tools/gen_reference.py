#!/usr/bin/env python3
"""Regenerate golden reference outputs for the hf-chat-template corpus.

Ground truth is ``transformers.apply_chat_template``. This script is a development-only tool:
it is **not** part of the published crate. The reference files it writes (``expected/*.txt``)
are committed and checked byte-for-byte by ``tests/m3_corpus.rs``, so CI never needs
``transformers`` installed — only regeneration does.

Setup (pin the version you regenerate with; record it in each model's PROVENANCE.md):

    python -m venv .venv && . .venv/bin/activate
    pip install transformers jinja2 sentencepiece protobuf
    python tools/gen_reference.py                       # all models
    python tools/gen_reference.py qwen2.5-0.5b-instruct # one model

Each ``tests/corpus/<model>/`` directory must contain:

    meta.json              {"model_id": "...", "revision": "...", ...}
    tokenizer_config.json  trimmed config (the source the Rust side renders from)
    cases/<name>.json      {"template_name": null | "<name>", "input": <RenderInput>}

and this script writes ``expected/<name>.txt`` for each case.

Determinism: the generic corpus deliberately avoids ``strftime_now`` (date-dependent) models;
add date-pinning here before introducing them.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import transformers
from transformers import AutoTokenizer

CORPUS = Path(__file__).resolve().parent.parent / "tests" / "corpus"

# Keys of a case's ``input`` that map to explicit apply_chat_template parameters; anything else
# is forwarded as a keyword (e.g. enable_thinking, builtin_tools).
RESERVED = {"messages", "tools", "documents", "add_generation_prompt"}


def render_case(tok, case: dict) -> str:
    inp = case["input"]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": inp.get("add_generation_prompt", False),
    }
    if inp.get("tools"):
        kwargs["tools"] = inp["tools"]
    if inp.get("documents"):
        kwargs["documents"] = inp["documents"]
    if case.get("template_name"):
        kwargs["chat_template"] = case["template_name"]
    for key, value in inp.items():
        if key not in RESERVED:
            kwargs[key] = value
    return tok.apply_chat_template(inp["messages"], **kwargs)


@contextmanager
def clock_freeze(meta: dict):
    """If the model pins ``clock_unix_secs``, freeze ``strftime_now`` to that instant so date-stamped
    templates are reproducible. transformers' ``strftime_now`` is a closure calling ``datetime.now()``
    bound to ``chat_template_utils.datetime`` (``utils/chat_template_utils.py``), so we swap just that
    one attribute — robust, and it avoids freezegun's sys.modules walk (which force-imports
    transformers' lazy, optional-dependency submodules and crashes). The instant is the UTC civil
    time of the Unix seconds, which the Rust runner reproduces with ``FixedClock::from_unix_secs``
    (also UTC), keeping both sides byte-identical."""
    secs = meta.get("clock_unix_secs")
    if secs is None:
        yield
        return
    from transformers.utils import chat_template_utils as ctu

    frozen = datetime.fromtimestamp(secs, tz=timezone.utc).replace(tzinfo=None)
    real = ctu.datetime

    class _FrozenDateTime(real):
        @classmethod
        def now(cls, tz=None):
            return frozen if tz is None else frozen.replace(tzinfo=tz)

    ctu.datetime = _FrozenDateTime
    try:
        yield
    finally:
        ctu.datetime = real


def gen_model(mdir: Path) -> None:
    meta = json.loads((mdir / "meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(meta["model_id"], revision=meta.get("revision"))
    out_dir = mdir / "expected"
    out_dir.mkdir(exist_ok=True)
    with clock_freeze(meta):
        for case_file in sorted((mdir / "cases").glob("*.json")):
            case = json.loads(case_file.read_text())
            rendered = render_case(tok, case)
            # Write exact bytes; .gitattributes marks these files -text so git won't normalize them.
            (out_dir / f"{case_file.stem}.txt").write_text(rendered, encoding="utf-8", newline="")
            print(f"  {mdir.name}/{case_file.stem}: {len(rendered)} chars")


def main() -> None:
    print(f"transformers {transformers.__version__}")
    selected = sys.argv[1:] or [p.name for p in sorted(CORPUS.iterdir()) if p.is_dir()]
    for name in selected:
        mdir = CORPUS / name
        if not (mdir / "meta.json").exists():
            print(f"skip {name}: no meta.json")
            continue
        print(f"# {name}")
        gen_model(mdir)


if __name__ == "__main__":
    main()
