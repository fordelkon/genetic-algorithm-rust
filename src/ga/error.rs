use std::fmt::{Display, Formatter};
/// Error definitions used across the GA engine.

/// Recoverable errors returned by GA APIs.
#[derive(Debug, Clone, PartialEq)]
pub enum GaError {
    /// Configuration values are invalid or inconsistent.
    InvalidConfig(String),

    /// A population-dependent operation was called on an empty population.
    EmptyPopulation,

    /// Fitness was accessed before evaluation.
    UnevaluatedFitness,

    /// The requested gene scalar type is not supported by this crate.
    UnsupportedGeneType(String),

    /// Reporting or visualization failed.
    Visualization(String),
}

impl Display for GaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid config: {message}"),
            Self::EmptyPopulation => write!(f, "population is empty"),
            Self::UnevaluatedFitness => write!(f, "fitness is not evaluated"),
            Self::UnsupportedGeneType(name) => write!(f, "unsupported gene type: {name}"),
            Self::Visualization(message) => write!(f, "visualization error: {message}"),
        }
    }
}

impl std::error::Error for GaError {}
