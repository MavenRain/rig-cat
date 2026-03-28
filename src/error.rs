//! Project-wide error type.

/// All errors in rig-cat.
#[derive(Debug)]
pub enum Error {
    /// HTTP request failed.
    Http(ureq::Error),
    /// JSON serialization/deserialization failed.
    Json(serde_json::Error),
    /// IO operation failed.
    Io(std::io::Error),
    /// Provider returned an error response.
    Provider { status: u16, message: String },
    /// Missing required configuration.
    Config { field: String },
    /// Embedding dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(e) => write!(f, "HTTP error: {e}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::Provider { status, message } => {
                write!(f, "provider error ({status}): {message}")
            }
            Self::Config { field } => write!(f, "missing config: {field}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Http(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Provider { .. } | Self::Config { .. } | Self::DimensionMismatch { .. } => None,
        }
    }
}

impl From<ureq::Error> for Error {
    fn from(e: ureq::Error) -> Self { Self::Http(e) }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self { Self::Json(e) }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}
