//! Completion model trait: the core LLM abstraction.
//!
//! A `CompletionModel` takes a prompt and returns a response,
//! wrapped in `Io` for composable effect handling.

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;
use serde::{Deserialize, Serialize};

use crate::error::Error;

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    role: Role,
    content: String,
}

impl Message {
    #[must_use]
    pub fn new(role: Role, content: String) -> Self {
        Self { role, content }
    }

    #[must_use]
    pub fn role(&self) -> &Role { &self.role }

    #[must_use]
    pub fn content(&self) -> &str { &self.content }

    #[must_use]
    pub fn system(content: String) -> Self {
        Self::new(Role::System, content)
    }

    #[must_use]
    pub fn user(content: String) -> Self {
        Self::new(Role::User, content)
    }

    #[must_use]
    pub fn assistant(content: String) -> Self {
        Self::new(Role::Assistant, content)
    }
}

/// The role of a message sender.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// A completion request.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    messages: Vec<Message>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
}

impl CompletionRequest {
    #[must_use]
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            temperature: None,
            max_tokens: None,
        }
    }

    #[must_use]
    pub fn messages(&self) -> &[Message] { &self.messages }

    #[must_use]
    pub fn temperature(&self) -> Option<f64> { self.temperature }

    #[must_use]
    pub fn max_tokens(&self) -> Option<u32> { self.max_tokens }

    #[must_use]
    pub fn with_temperature(self, t: f64) -> Self {
        Self { temperature: Some(t), ..self }
    }

    #[must_use]
    pub fn with_max_tokens(self, n: u32) -> Self {
        Self { max_tokens: Some(n), ..self }
    }
}

/// A completion response.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    content: String,
    model: String,
}

impl CompletionResponse {
    #[must_use]
    pub fn new(content: String, model: String) -> Self {
        Self { content, model }
    }

    #[must_use]
    pub fn content(&self) -> &str { &self.content }

    #[must_use]
    pub fn model(&self) -> &str { &self.model }
}

/// A token chunk from a streaming response.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    delta: String,
}

impl StreamChunk {
    #[must_use]
    pub fn new(delta: String) -> Self { Self { delta } }

    #[must_use]
    pub fn delta(&self) -> &str { &self.delta }
}

/// The core LLM abstraction: send a request, get a response.
///
/// All provider-specific logic is behind this trait.
/// Returns `Io<Error, _>` for composable effect handling.
pub trait CompletionModel {
    /// Send a completion request and get a full response.
    fn complete(&self, request: CompletionRequest) -> Io<Error, CompletionResponse>;

    /// Send a completion request and get a streaming response.
    fn stream(&self, request: CompletionRequest) -> Stream<Error, StreamChunk>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_constructors_set_correct_roles() {
        let sys = Message::system("sys".into());
        let usr = Message::user("usr".into());
        let ast = Message::assistant("ast".into());
        assert!(matches!(sys.role(), Role::System));
        assert!(matches!(usr.role(), Role::User));
        assert!(matches!(ast.role(), Role::Assistant));
        assert_eq!(sys.content(), "sys");
        assert_eq!(usr.content(), "usr");
        assert_eq!(ast.content(), "ast");
    }

    #[test]
    fn completion_request_builder_applies_options() {
        let req = CompletionRequest::new(vec![Message::user("hi".into())])
            .with_temperature(0.5)
            .with_max_tokens(100);
        assert_eq!(req.messages().len(), 1);
        assert!((req.temperature().unwrap_or(0.0) - 0.5).abs() < 1e-10);
        assert_eq!(req.max_tokens(), Some(100));
    }

    #[test]
    fn completion_request_defaults_are_none() {
        let req = CompletionRequest::new(vec![]);
        assert!(req.temperature().is_none());
        assert!(req.max_tokens().is_none());
    }
}
