//! Optional Hugging Face Hub integration (`hub` feature).
//!
//! Fetches a model's `tokenizer_config.json` straight from the Hub via the synchronous
//! (`ureq`) client, so callers can go from a repo id to a compiled template in one call.
//! Authentication follows `hf-hub`'s own discovery: the `HF_TOKEN` env var or the cached
//! token written by `huggingface-cli login`, which is what gated repos need.

use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};

use crate::config::TokenizerConfig;
use crate::error::Error;

/// Download and parse `tokenizer_config.json` for `repo_id` at `revision` (`None` = default
/// branch). Network, auth, and missing-file failures surface as [`Error::Hub`]; a malformed
/// config surfaces as [`Error::Config`].
pub(crate) fn fetch_config(
    repo_id: &str,
    revision: Option<&str>,
) -> Result<TokenizerConfig, Error> {
    let api = Api::new().map_err(|e| Error::Hub(e.to_string()))?;
    let repo = match revision {
        Some(rev) => api.repo(Repo::with_revision(
            repo_id.to_owned(),
            RepoType::Model,
            rev.to_owned(),
        )),
        None => api.model(repo_id.to_owned()),
    };
    let path = repo
        .get("tokenizer_config.json")
        .map_err(|e| Error::Hub(e.to_string()))?;
    let bytes = std::fs::read(&path).map_err(|e| Error::Hub(e.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::Config(e.to_string()))
}
