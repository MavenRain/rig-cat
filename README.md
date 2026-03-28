# rig-cat

An LLM agent framework built on [comp-cat-rs](https://github.com/MavenRain/comp-cat-rs).  No async, no tokio.  All effects are `Io<Error, A>`, all concurrency is `Fiber`, all streaming is `Stream`.

## Installation

```toml
[dependencies]
rig-cat = "0.1"
```

## Quick start

```rust
use rig_cat::provider::openai::{OpenAiCompletion, ApiKey, ModelName};
use rig_cat::agent::AgentBuilder;

let model = OpenAiCompletion::new(
    ApiKey::new("sk-...".into()),
    ModelName::new("gpt-4o".into()),
);

let agent = AgentBuilder::new(model)
    .preamble("You are a helpful assistant.")
    .temperature(0.7)
    .build();

// Nothing runs until .run()
let response = agent.prompt("What is a Kan extension?").run();
```

## Architecture

```
model/          CompletionModel trait, Message, Request/Response types
embedding/      EmbeddingModel trait, Embedding type, cosine similarity
tool/           Tool trait, generic Toolbox<T>
vector_store/   VectorStoreIndex trait, InMemoryVectorStore
agent/          Agent<M, T> with builder pattern
pipeline/       RAG pipeline via Io::flat_map chains
provider/       OpenAI, Anthropic implementations
error/          Hand-rolled Error enum
```

Every function that touches the network returns `Io<Error, A>`.  Composition happens via `map`, `flat_map`, `zip`.  Side effects only happen when you call `.run()`.

## Providers

### OpenAI

```rust
use rig_cat::provider::openai::{OpenAiCompletion, OpenAiEmbedding, ApiKey, ModelName};

let completion = OpenAiCompletion::new(
    ApiKey::new(api_key),
    ModelName::new("gpt-4o".into()),
);

let embedding = OpenAiEmbedding::new(
    ApiKey::new(api_key),
    ModelName::new("text-embedding-3-small".into()),
);
```

### Anthropic

```rust
use rig_cat::provider::anthropic::{AnthropicCompletion, ApiKey, ModelName};

let model = AnthropicCompletion::new(
    ApiKey::new(api_key),
    ModelName::new("claude-sonnet-4-20250514".into()),
    4096, // max_tokens
);
```

## Tools

Tools are generic over a concrete type.  For heterogeneous tools, define an enum:

```rust
use rig_cat::tool::{Tool, ToolDefinition, Toolbox};
use comp_cat_rs::effect::io::Io;
use serde_json::Value;

struct Calculator;

impl Tool for Calculator {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new(
            "calculate".into(),
            "Evaluate a math expression".into(),
            serde_json::json!({"type": "object", "properties": {"expr": {"type": "string"}}}),
        )
    }

    fn call(&self, args: Value) -> Io<rig_cat::error::Error, Value> {
        Io::pure(serde_json::json!({"result": 42}))
    }
}

let toolbox = Toolbox::new().with_tool(Calculator);
let result = toolbox.invoke("calculate", serde_json::json!({"expr": "6 * 7"})).run();
```

## RAG pipeline

The `pipeline::rag` function composes embedding, search, and generation into a single `Io` chain:

```rust
use std::rc::Rc;
use rig_cat::pipeline::rag;

let response = rag(
    "What is the return policy?".into(),
    Rc::new(completion_model),
    Rc::new(embedding_model),
    Rc::new(vector_store),
    Some("You are a customer service agent.".into()),
    3, // top_k
).run();
```

## Concurrency

Use `comp-cat-rs` `Fiber` for parallel LLM calls:

```rust
use comp_cat_rs::effect::fiber::par_zip;

let task_a = agent.prompt("Summarize document A");
let task_b = agent.prompt("Summarize document B");

// Both run on separate threads
let (summary_a, summary_b) = par_zip(task_a, task_b).run()?;
```

## Why no async?

LLM API calls are high-latency (1-30 seconds) and low-concurrency (a handful of calls, not thousands).  Thread-per-request via `Fiber` is perfectly adequate.  The benefit: no tokio, no `Pin<Box<dyn Future>>`, no colored functions.  Everything composes with `flat_map`.

## The categorical foundation

This crate is the practical application of the thesis proved in [comp-cat-theory](https://github.com/MavenRain/comp-cat-theory) (Lean 4):

- `Io` is a monad, which is a pair of Kan extensions
- `Stream` is a colimit, which is a left Kan extension
- `Fiber::fork` is a coproduct, `Fiber::join` is a limit
- The RAG pipeline is a composition of monadic effects

The proofs are in Lean 4 with zero `sorry`s.  The Rust code is the runtime implementation.

## Status

Alpha.  The core architecture is stable, but:

- Streaming responses are not yet implemented (providers return `Stream::empty()`)
- Tool-calling loop (agent calls tool, feeds result back, repeats) is not yet implemented
- Only two providers (OpenAI, Anthropic); more planned
- Only in-memory vector store; external stores planned

## License

MIT
