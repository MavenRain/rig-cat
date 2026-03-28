//! `OpenAI` provider: completion and embedding models.

use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;
use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::model::{
    CompletionModel, CompletionRequest, CompletionResponse, StreamChunk,
};
use crate::embedding::{Embedding, EmbeddingModel, EmbeddingRequest};

/// Newtype for the `OpenAI` API key.
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

/// `OpenAI` completion model.
pub struct OpenAiCompletion {
    api_key: ApiKey,
    model: ModelName,
}

impl OpenAiCompletion {
    #[must_use]
    pub fn new(api_key: ApiKey, model: ModelName) -> Self {
        Self { api_key, model }
    }
}

/// `OpenAI` embedding model.
pub struct OpenAiEmbedding {
    api_key: ApiKey,
    model: ModelName,
}

impl OpenAiEmbedding {
    #[must_use]
    pub fn new(api_key: ApiKey, model: ModelName) -> Self {
        Self { api_key, model }
    }
}

// --- Request/response JSON shapes ---

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    model: String,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: Option<String>,
}

#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f64>,
}

// --- Trait impls ---

impl CompletionModel for OpenAiCompletion {
    fn complete(&self, request: CompletionRequest) -> Io<Error, CompletionResponse> {
        let api_key = self.api_key.clone();
        let model_name = self.model.clone();
        Io::suspend(move || {
            let messages: Vec<ChatMessage> = request.messages().iter().map(|m| {
                ChatMessage {
                    role: match m.role() {
                        crate::model::Role::System => "system".to_owned(),
                        crate::model::Role::User => "user".to_owned(),
                        crate::model::Role::Assistant => "assistant".to_owned(),
                    },
                    content: m.content().to_owned(),
                }
            }).collect();

            let body = ChatRequest {
                model: model_name.as_str().to_owned(),
                messages,
                temperature: request.temperature(),
                max_tokens: request.max_tokens(),
            };

            let resp: ChatResponse = ureq::post("https://api.openai.com/v1/chat/completions")
                .header("Authorization", &format!("Bearer {}", api_key.as_str()))
                .header("Content-Type", "application/json")
                .send_json(&body)
                .map_err(Error::from)?
                .into_body()
                .read_json()
                .map_err(Error::from)?;

            let content = resp.choices.first()
                .and_then(|c| c.message.content.clone())
                .unwrap_or_default();

            Ok(CompletionResponse::new(content, resp.model))
        })
    }

    fn stream(&self, _request: CompletionRequest) -> Stream<Error, StreamChunk> {
        // TODO: implement SSE streaming
        Stream::empty()
    }
}

impl EmbeddingModel for OpenAiEmbedding {
    fn embed(&self, request: EmbeddingRequest) -> Io<Error, Vec<Embedding>> {
        let api_key = self.api_key.clone();
        let model_name = self.model.clone();
        Io::suspend(move || {
            let body = EmbedRequest {
                model: model_name.as_str().to_owned(),
                input: request.texts().to_vec(),
            };

            let resp: EmbedResponse = ureq::post("https://api.openai.com/v1/embeddings")
                .header("Authorization", &format!("Bearer {}", api_key.as_str()))
                .header("Content-Type", "application/json")
                .send_json(&body)
                .map_err(Error::from)?
                .into_body()
                .read_json()
                .map_err(Error::from)?;

            Ok(resp.data.into_iter()
                .map(|d| Embedding::new(d.embedding))
                .collect())
        })
    }
}
