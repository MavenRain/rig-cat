//! Pipeline: composable RAG and multi-step workflows.
//!
//! A pipeline chains retrieval, augmentation, and generation
//! steps using `Io::flat_map`.  Each step is an `Io<Error, _>`,
//! composed via monadic sequencing.

use std::rc::Rc;

use comp_cat_rs::effect::io::Io;

use crate::embedding::{Embedding, EmbeddingModel, EmbeddingRequest};
use crate::error::Error;
use crate::model::CompletionModel;
use crate::vector_store::{SearchResult, VectorStoreIndex};

/// Run a RAG pipeline: embed the query, search the store,
/// augment the prompt with context, and generate a response.
///
/// This is the canonical Retrieval-Augmented Generation pattern,
/// expressed as a single `Io::flat_map` chain.
///
/// Takes `Rc`-wrapped references so the closures can capture
/// them across `flat_map` boundaries without `'static` ownership issues.
#[allow(clippy::needless_pass_by_value)] // Rc's are moved into 'static closures
pub fn rag<M, E, V>(
    query: String,
    model: Rc<M>,
    embedder: Rc<E>,
    store: Rc<V>,
    preamble: Option<String>,
    top_k: usize,
) -> Io<Error, String>
where
    M: CompletionModel + 'static,
    E: EmbeddingModel + 'static,
    V: VectorStoreIndex + 'static,
{
    let embed_io = embedder.embed(EmbeddingRequest::single(query.clone()));
    embed_io.flat_map(move |embeddings| {
        let query_embedding: Embedding = embeddings.into_iter()
            .next()
            .unwrap_or_else(|| Embedding::new(Vec::new()));
        store.search(&query_embedding, top_k).flat_map(move |results| {
            let context = format_context(&results);
            let augmented_prompt = format!(
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above."
            );
            let messages = preamble.iter()
                .map(|p| crate::model::Message::system(p.clone()))
                .chain(std::iter::once(crate::model::Message::user(augmented_prompt)))
                .collect();
            let request = crate::model::CompletionRequest::new(messages);
            model.complete(request).map(|r| r.content().to_owned())
        })
    })
}

fn format_context(results: &[SearchResult]) -> String {
    results.iter()
        .enumerate()
        .map(|(i, r)| format!("[{}] (score: {:.3}) {}", i + 1, r.score(), r.document().content()))
        .collect::<Vec<_>>()
        .join("\n")
}
