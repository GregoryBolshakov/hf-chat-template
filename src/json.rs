//! Python-compatible `tojson` filter.
//!
//! M0 confirmed minijinja's built-in `tojson` diverges from `transformers`: it sorts object
//! keys and omits separators spaces. `transformers` uses `json.dumps(x, ensure_ascii=False)`,
//! which preserves **insertion order** and uses `", "` / `": "` separators. We register our
//! own filter to match Python byte-for-byte.
//!
//! - No `indent`  → compact with Python's default separators `(", ", ": ")`.
//! - `indent=N`   → pretty-printed with N-space indent and `": "` key separator
//!   (matching `json.dumps(x, indent=N, ensure_ascii=False)`).
//! - `ensure_ascii=False` always (transformers' default for chat templates).

use minijinja::value::{Kwargs, Value};
use minijinja::{Error, ErrorKind};
use serde::Serialize;
use serde_json::ser::Formatter;
use std::io;

/// Compact formatter matching Python's `json.dumps` default separators `(", ", ": ")`.
struct PythonCompact;

impl Formatter for PythonCompact {
    #[inline]
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
}

/// Serialize a minijinja [`Value`] to a Python-`json.dumps`-compatible string.
///
/// `indent`: `None` → compact; `Some(n)` → pretty with `n`-space indent.
pub(crate) fn python_tojson(value: &Value, indent: Option<usize>) -> Result<String, Error> {
    let mut buf: Vec<u8> = Vec::with_capacity(64);
    match indent {
        None => {
            let mut ser = serde_json::Serializer::with_formatter(&mut buf, PythonCompact);
            value
                .serialize(&mut ser)
                .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
        }
        Some(n) => {
            let indent_bytes = vec![b' '; n];
            let pretty = serde_json::ser::PrettyFormatter::with_indent(&indent_bytes);
            let mut ser = serde_json::Serializer::with_formatter(&mut buf, pretty);
            value
                .serialize(&mut ser)
                .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
        }
    }
    // serde_json only emits valid UTF-8.
    Ok(String::from_utf8(buf).expect("serde_json emits valid UTF-8"))
}

/// The `tojson` filter as registered on the environment: `value | tojson` or
/// `value | tojson(indent=2)`.
pub(crate) fn tojson_filter(value: Value, kwargs: Kwargs) -> Result<String, Error> {
    let indent: Option<usize> = kwargs.get("indent").unwrap_or(None);
    // Be lenient about extra kwargs (e.g. ensure_ascii) that some templates pass; we don't
    // call kwargs.assert_all_used().
    python_tojson(&value, indent)
}
