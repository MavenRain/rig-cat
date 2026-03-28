//! Vector store trait and in-memory implementation.

use comp_cat_rs::effect::io::Io;

use crate::embedding::{Embedding, EmbeddingModel, EmbeddingRequest};
use crate::error::Error;

/// A document stored in a vector store.
#[derive(Debug, Clone)]
pub struct Document {
    id: String,
    content: String,
    embedding: Embedding,
}

impl Document {
    #[must_use]
    pub fn new(id: String, content: String, embedding: Embedding) -> Self {
        Self { id, content, embedding }
    }

    #[must_use]
    pub fn id(&self) -> &str { &self.id }

    #[must_use]
    pub fn content(&self) -> &str { &self.content }

    #[must_use]
    pub fn embedding(&self) -> &Embedding { &self.embedding }
}

/// A search result: a document with its similarity score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    document: Document,
    score: f64,
}

impl SearchResult {
    #[must_use]
    pub fn new(document: Document, score: f64) -> Self {
        Self { document, score }
    }

    #[must_use]
    pub fn document(&self) -> &Document { &self.document }

    #[must_use]
    pub fn score(&self) -> f64 { self.score }
}

/// A vector store index: store documents, search by similarity.
pub trait VectorStoreIndex {
    /// Search for the top-k most similar documents to the query.
    fn search(&self, query: &Embedding, top_k: usize) -> Io<Error, Vec<SearchResult>>;
}

/// In-memory vector store: stores documents in a Vec,
/// searches by cosine similarity.
pub struct InMemoryVectorStore {
    documents: Vec<Document>,
}

impl InMemoryVectorStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self { Self { documents: Vec::new() } }

    /// Add documents to the store.
    #[must_use]
    pub fn with_documents(self, docs: Vec<Document>) -> Self {
        Self {
            documents: self.documents.into_iter().chain(docs).collect(),
        }
    }

    /// Ingest raw texts: embed them and store.
    pub fn ingest<M: EmbeddingModel>(
        texts: &[(String, String)],
        model: &M,
    ) -> Io<Error, Self> {
        let contents: Vec<String> = texts.iter().map(|(_, c)| c.clone()).collect();
        let ids: Vec<String> = texts.iter().map(|(id, _)| id.clone()).collect();
        model.embed(EmbeddingRequest::new(contents.clone())).map(move |embeddings| {
            let docs = ids.into_iter()
                .zip(contents)
                .zip(embeddings)
                .map(|((id, content), emb)| Document::new(id, content, emb))
                .collect();
            Self { documents: Vec::new() }.with_documents(docs)
        })
    }
}

impl Default for InMemoryVectorStore {
    fn default() -> Self { Self::new() }
}

impl VectorStoreIndex for InMemoryVectorStore {
    fn search(&self, query: &Embedding, top_k: usize) -> Io<Error, Vec<SearchResult>> {
        let results: Result<Vec<SearchResult>, Error> = self.documents.iter()
            .map(|doc| {
                doc.embedding().cosine_similarity(query)
                    .map(|score| SearchResult::new(doc.clone(), score))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|scored| {
                // Insert into a sorted vec via fold (no mut needed)
                scored.into_iter()
                    .fold(Vec::<SearchResult>::new(), |acc, result| {
                        let score = result.score();
                        let pos = acc.iter()
                            .position(|r| r.score() < score)
                            .unwrap_or(acc.len());
                        let (head, tail) = (
                            acc.iter().take(pos).cloned().collect::<Vec<_>>(),
                            acc.iter().skip(pos).cloned().collect::<Vec<_>>(),
                        );
                        head.into_iter()
                            .chain(std::iter::once(result))
                            .chain(tail)
                            .take(top_k)
                            .collect()
                    })
            });
        Io::suspend(move || results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, emb: Vec<f64>) -> Document {
        Document::new(id.into(), format!("content of {id}"), Embedding::new(emb))
    }

    #[test]
    fn search_returns_most_similar_first() -> Result<(), Error> {
        let store = InMemoryVectorStore::new().with_documents(vec![
            make_doc("far", vec![0.0, 1.0]),
            make_doc("close", vec![1.0, 0.1]),
            make_doc("mid", vec![0.7, 0.7]),
        ]);
        let query = Embedding::new(vec![1.0, 0.0]);
        let results = store.search(&query, 3).run()?;
        assert_eq!(results.first().map(|r| r.document().id()), Some("close"));
        Ok(())
    }

    #[test]
    fn search_respects_top_k() -> Result<(), Error> {
        let store = InMemoryVectorStore::new().with_documents(vec![
            make_doc("a", vec![1.0, 0.0]),
            make_doc("b", vec![0.9, 0.1]),
            make_doc("c", vec![0.0, 1.0]),
        ]);
        let query = Embedding::new(vec![1.0, 0.0]);
        let results = store.search(&query, 1).run()?;
        assert_eq!(results.len(), 1);
        Ok(())
    }

    #[test]
    fn search_empty_store_returns_empty() -> Result<(), Error> {
        let store = InMemoryVectorStore::new();
        let query = Embedding::new(vec![1.0, 0.0]);
        let results = store.search(&query, 5).run()?;
        assert!(results.is_empty());
        Ok(())
    }
}
