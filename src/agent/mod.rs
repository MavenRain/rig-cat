//! Agent: the high-level LLM abstraction.
//!
//! An Agent composes a completion model, a system preamble,
//! optional tools, and an optional vector store into a single
//! `.prompt()` call that returns `Io<Error, String>`.

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;

use crate::error::Error;
use crate::model::{
    CompletionModel, CompletionRequest, Message, StreamChunk,
};
use crate::tool::{Tool, Toolbox};

/// An agent: a configured LLM with preamble, tools, and context.
///
/// Built via `AgentBuilder`.  Generic over:
/// - `M`: the completion model
/// - `T`: the tool type (use an enum for heterogeneous tools)
pub struct Agent<M: CompletionModel, T: Tool> {
    model: M,
    preamble: Option<String>,
    tools: Toolbox<T>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
}

/// Builder for constructing an `Agent`.
pub struct AgentBuilder<M: CompletionModel, T: Tool> {
    model: M,
    preamble: Option<String>,
    tools: Toolbox<T>,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
}

impl<M: CompletionModel, T: Tool> AgentBuilder<M, T> {
    /// Start building an agent with the given model.
    #[must_use]
    pub fn new(model: M) -> Self {
        Self {
            model,
            preamble: None,
            tools: Toolbox::new(),
            temperature: None,
            max_tokens: None,
        }
    }

    /// Set the system preamble.
    #[must_use]
    pub fn preamble(self, preamble: impl Into<String>) -> Self {
        Self { preamble: Some(preamble.into()), ..self }
    }

    /// Set the tools available to the agent.
    #[must_use]
    pub fn tools(self, tools: Toolbox<T>) -> Self {
        Self { tools, ..self }
    }

    /// Set the temperature.
    #[must_use]
    pub fn temperature(self, t: f64) -> Self {
        Self { temperature: Some(t), ..self }
    }

    /// Set the max tokens.
    #[must_use]
    pub fn max_tokens(self, n: u32) -> Self {
        Self { max_tokens: Some(n), ..self }
    }

    /// Build the agent.
    #[must_use]
    pub fn build(self) -> Agent<M, T> {
        Agent {
            model: self.model,
            preamble: self.preamble,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
        }
    }
}

impl<M: CompletionModel, T: Tool> Agent<M, T> {
    /// Send a prompt and get a complete response.
    pub fn prompt(&self, user_input: &str) -> Io<Error, String> {
        let request = self.build_request(user_input);
        self.model.complete(request).map(|r| r.content().to_owned())
    }

    /// Send a prompt and get a streaming response.
    pub fn prompt_stream(&self, user_input: &str) -> Stream<Error, StreamChunk> {
        let request = self.build_request(user_input);
        self.model.stream(request)
    }

    /// Access the toolbox.
    #[must_use]
    pub fn tools(&self) -> &Toolbox<T> { &self.tools }

    fn build_request(&self, user_input: &str) -> CompletionRequest {
        let messages = self.preamble.iter()
            .map(|p| Message::system(p.clone()))
            .chain(std::iter::once(Message::user(user_input.to_owned())))
            .collect();
        let request = CompletionRequest::new(messages);
        let request = match self.temperature {
            Some(t) => request.with_temperature(t),
            None => request,
        };
        match self.max_tokens {
            Some(n) => request.with_max_tokens(n),
            None => request,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolDefinition;

    struct FakeModel;

    impl CompletionModel for FakeModel {
        fn complete(&self, request: CompletionRequest) -> Io<Error, crate::model::CompletionResponse> {
            let content = request.messages().iter()
                .map(|m| m.content().to_owned())
                .collect::<Vec<_>>()
                .join("|");
            Io::pure(crate::model::CompletionResponse::new(content, "fake".into()))
        }

        fn stream(&self, _request: CompletionRequest) -> Stream<Error, StreamChunk> {
            Stream::empty()
        }
    }

    struct FakeTool;

    impl Tool for FakeTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("fake".into(), "fake".into(), serde_json::json!({}))
        }
        fn call(&self, _args: serde_json::Value) -> Io<Error, serde_json::Value> {
            Io::pure(serde_json::json!({}))
        }
    }

    #[test]
    fn agent_includes_preamble_in_request() -> Result<(), Error> {
        let agent: Agent<FakeModel, FakeTool> = AgentBuilder::new(FakeModel)
            .preamble("You are helpful.")
            .build();
        let response = agent.prompt("hello").run()?;
        assert!(response.contains("You are helpful."));
        assert!(response.contains("hello"));
        Ok(())
    }

    #[test]
    fn agent_without_preamble_sends_only_user_message() -> Result<(), Error> {
        let agent: Agent<FakeModel, FakeTool> = AgentBuilder::new(FakeModel).build();
        let response = agent.prompt("hello").run()?;
        assert_eq!(response, "hello");
        Ok(())
    }

    #[test]
    fn agent_applies_temperature_and_max_tokens() {
        let agent: Agent<FakeModel, FakeTool> = AgentBuilder::new(FakeModel)
            .temperature(0.5)
            .max_tokens(100)
            .build();
        let request = agent.build_request("test");
        assert!((request.temperature().unwrap_or(0.0) - 0.5).abs() < 1e-10);
        assert_eq!(request.max_tokens(), Some(100));
    }
}
