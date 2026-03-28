//! # rig-cat
//!
//! An LLM agent framework built on `comp-cat-rs`.
//!
//! No async, no tokio.  All effects are `Io<Error, A>`, all
//! concurrency is `Fiber`, all streaming is `Stream`.
//! The entire runtime is synchronous composition with
//! thread-based parallelism when needed.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use rig_cat::provider::openai::{OpenAiCompletion, ApiKey, ModelName};
//! use rig_cat::agent::AgentBuilder;
//!
//! let model = OpenAiCompletion::new(
//!     ApiKey::new(std::env::var("OPENAI_API_KEY")?),
//!     ModelName::new("gpt-4o".into()),
//! );
//!
//! let agent = AgentBuilder::new(model)
//!     .preamble("You are a helpful assistant.")
//!     .temperature(0.7)
//!     .build();
//!
//! let response = agent.prompt("What is a Kan extension?").run()?;
//! println!("{response}");
//! ```

pub mod error;
pub mod model;
pub mod embedding;
pub mod tool;
pub mod vector_store;
pub mod agent;
pub mod provider;
pub mod pipeline;
