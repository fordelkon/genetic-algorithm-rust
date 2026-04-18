use crate::ga::engine::config::OptimizationMode;
use crate::ga::error::GaError;

/// Unified evaluation payload for single- and multi-objective optimization.
#[derive(Debug, Clone, PartialEq)]
pub enum Evaluation {
    /// Scalar score used by legacy single-objective GA workflows.
    Single(f64),
    /// Objective vector used by NSGA-II, interpreted as minimization targets.
    Multi(Vec<f64>),
}

impl Evaluation {
    /// Returns the scalar fitness value when this is a single-objective evaluation.
    pub fn as_single(&self) -> Result<f64, GaError> {
        match self {
            Self::Single(value) => Ok(*value),
            Self::Multi(_) => Err(GaError::UnsupportedOperation(
                "single-objective fitness is unavailable in NSGA-II mode".into(),
            )),
        }
    }

    /// Returns the objective slice when this is a multi-objective evaluation.
    pub fn as_multi(&self) -> Result<&[f64], GaError> {
        match self {
            Self::Single(_) => Err(GaError::UnsupportedOperation(
                "multi-objective values are unavailable in single-objective mode".into(),
            )),
            Self::Multi(values) => Ok(values),
        }
    }

    /// Validates that this evaluation matches the configured optimization mode.
    pub fn validate_for_mode(&self, optimization_mode: &OptimizationMode) -> Result<(), GaError> {
        match (optimization_mode, self) {
            (OptimizationMode::SingleObjective, Evaluation::Single(_)) => Ok(()),
            (OptimizationMode::SingleObjective, Evaluation::Multi(_)) => {
                Err(GaError::InvalidConfig(
                    "single-objective mode requires evaluators that return a scalar fitness".into(),
                ))
            }
            (OptimizationMode::Nsga2 { num_objectives }, Evaluation::Multi(values))
                if values.len() == *num_objectives =>
            {
                Ok(())
            }
            (OptimizationMode::Nsga2 { num_objectives }, Evaluation::Multi(values)) => {
                Err(GaError::InvalidConfig(format!(
                    "NSGA-II evaluator returned {} objectives, expected {}",
                    values.len(),
                    num_objectives
                )))
            }
            (OptimizationMode::Nsga2 { .. }, Evaluation::Single(_)) => Err(GaError::InvalidConfig(
                "NSGA-II mode requires evaluators that return objective vectors".into(),
            )),
        }
    }
}

/// Converts evaluator return values into the unified [`Evaluation`] enum.
pub trait IntoEvaluation {
    fn into_evaluation(self) -> Evaluation;
}

impl IntoEvaluation for Evaluation {
    fn into_evaluation(self) -> Evaluation {
        self
    }
}

impl IntoEvaluation for f64 {
    fn into_evaluation(self) -> Evaluation {
        Evaluation::Single(self)
    }
}

impl IntoEvaluation for Vec<f64> {
    fn into_evaluation(self) -> Evaluation {
        Evaluation::Multi(self)
    }
}
