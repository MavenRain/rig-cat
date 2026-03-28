//! Embedding model trait and vector types.

use comp_cat_rs::effect::io::Io;

use crate::error::Error;

/// A dense vector embedding.
#[derive(Debug, Clone)]
pub struct Embedding {
    values: Vec<f64>,
}

impl Embedding {
    #[must_use]
    pub fn new(values: Vec<f64>) -> Self { Self { values } }

    #[must_use]
    pub fn values(&self) -> &[f64] { &self.values }

    #[must_use]
    pub fn dimension(&self) -> usize { self.values.len() }

    /// Cosine similarity between two embeddings.
    ///
    /// # Errors
    ///
    /// Returns `Error::DimensionMismatch` if dimensions differ.
    pub fn cosine_similarity(&self, other: &Self) -> Result<f64, Error> {
        if self.dimension() == other.dimension() {
            let dot: f64 = self.values.iter()
                .zip(other.values.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm_a: f64 = self.values.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_b: f64 = other.values.iter().map(|x| x * x).sum::<f64>().sqrt();
            let denom = norm_a * norm_b;
            Ok(if denom == 0.0 { 0.0 } else { dot / denom })
        } else {
            Err(Error::DimensionMismatch {
                expected: self.dimension(),
                got: other.dimension(),
            })
        }
    }
}

/// An embedding request: one or more texts to embed.
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    texts: Vec<String>,
}

impl EmbeddingRequest {
    #[must_use]
    pub fn new(texts: Vec<String>) -> Self { Self { texts } }

    #[must_use]
    pub fn single(text: String) -> Self { Self { texts: vec![text] } }

    #[must_use]
    pub fn texts(&self) -> &[String] { &self.texts }
}

/// The core embedding abstraction: send text, get vectors.
pub trait EmbeddingModel {
    /// Embed one or more texts.
    fn embed(&self, request: EmbeddingRequest) -> Io<Error, Vec<Embedding>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_similarity_one() -> Result<(), Error> {
        let a = Embedding::new(vec![1.0, 0.0, 0.0]);
        let b = Embedding::new(vec![1.0, 0.0, 0.0]);
        let sim = a.cosine_similarity(&b)?;
        assert!((sim - 1.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() -> Result<(), Error> {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![0.0, 1.0]);
        let sim = a.cosine_similarity(&b)?;
        assert!(sim.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn opposite_vectors_have_similarity_negative_one() -> Result<(), Error> {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![-1.0, 0.0]);
        let sim = a.cosine_similarity(&b)?;
        assert!((sim + 1.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![1.0, 0.0, 0.0]);
        assert!(a.cosine_similarity(&b).is_err());
    }

    #[test]
    fn zero_vector_similarity_is_zero() -> Result<(), Error> {
        let a = Embedding::new(vec![0.0, 0.0]);
        let b = Embedding::new(vec![1.0, 0.0]);
        let sim = a.cosine_similarity(&b)?;
        assert!(sim.abs() < 1e-10);
        Ok(())
    }
}
