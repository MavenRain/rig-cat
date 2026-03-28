//! Anthropic provider: completion model.

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::model::{
    CompletionModel, CompletionRequest, CompletionResponse, StreamChunk,
};

/// Newtype for the Anthropic API key.
#[derive(Clone)]
pub struct ApiKey(String);

impl ApiKey {
    #[must_use]
    pub fn new(key: String) -> Self { Self(key) }

    fn as_str(&self) -> &str { &self.0 }
}

/// Newtype for a model name.
#[derive(Clone)]
pub struct ModelName(String);

impl ModelName {
    #[must_use]
    pub fn new(name: String) -> Self { Self(name) }

    fn as_str(&self) -> &str { &self.0 }
}

/// Anthropic completion model.
pub struct AnthropicCompletion {
    api_key: ApiKey,
    model: ModelName,
    max_tokens: u32,
}

impl AnthropicCompletion {
    #[must_use]
    pub fn new(api_key: ApiKey, model: ModelName, max_tokens: u32) -> Self {
        Self { api_key, model, max_tokens }
    }
}

// --- Request/response JSON shapes ---

#[derive(Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    model: String,
}

#[derive(Deserialize)]
struct ContentBlock {
    text: Option<String>,
}

// --- Trait impl ---

impl CompletionModel for AnthropicCompletion {
    fn complete(&self, request: CompletionRequest) -> Io<Error, CompletionResponse> {
        let api_key = self.api_key.clone();
        let model_name = self.model.clone();
        let default_max = self.max_tokens;
        Io::suspend(move || {
            let system_msg = request.messages().iter()
                .find(|m| matches!(m.role(), crate::model::Role::System))
                .map(|m| m.content().to_owned());

            let messages: Vec<AnthropicMessage> = request.messages().iter()
                .filter(|m| !matches!(m.role(), crate::model::Role::System))
                .map(|m| AnthropicMessage {
                    role: match m.role() {
                        crate::model::Role::Assistant => "assistant".to_owned(),
                        crate::model::Role::User | crate::model::Role::System => "user".to_owned(),
                    },
                    content: m.content().to_owned(),
                })
                .collect();

            let body = MessagesRequest {
                model: model_name.as_str().to_owned(),
                max_tokens: request.max_tokens().unwrap_or(default_max),
                system: system_msg,
                messages,
                temperature: request.temperature(),
            };

            let resp: MessagesResponse = ureq::post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", api_key.as_str())
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .send_json(&body)
                .map_err(Error::from)?
                .into_body()
                .read_json()
                .map_err(Error::from)?;

            let content: String = resp.content.iter()
                .filter_map(|b| b.text.clone())
                .collect();

            Ok(CompletionResponse::new(content, resp.model))
        })
    }

    fn stream(&self, _request: CompletionRequest) -> Stream<Error, StreamChunk> {
        // TODO: implement SSE streaming
        Stream::empty()
    }
}
