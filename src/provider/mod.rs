//! LLM provider implementations.
//!
//! Each submodule implements `CompletionModel` and/or `EmbeddingModel`
//! for a specific provider, using `ureq` for blocking HTTP and
//! `Io::suspend` for effect wrapping.

pub mod openai;
pub mod anthropic;
