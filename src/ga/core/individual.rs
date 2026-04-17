use crate::ga::core::evaluation::Evaluation;
use crate::ga::core::gene::GeneValue;
/// Individual representation for GA evolution.

/// A candidate solution with genes and optional fitness.
#[derive(Debug, Clone, PartialEq)]
pub struct Individual {
    /// Chromosome values.
    pub genes: Vec<GeneValue>,
    /// Unified evaluation payload, present after evaluation.
    pub evaluation: Option<Evaluation>,
    /// Pareto front rank used by NSGA-II.
    pub rank: Option<usize>,
    /// Crowding distance used by NSGA-II.
    pub crowding_distance: Option<f64>,
}

impl Individual {
    /// Creates a new unevaluated individual.
    pub fn new(genes: Vec<GeneValue>) -> Self {
        Self {
            genes,
            evaluation: None,
            rank: None,
            crowding_distance: None,
        }
    }

    /// Creates an individual with a known fitness value.
    pub fn with_fitness(genes: Vec<GeneValue>, fitness: f64) -> Self {
        Self {
            genes,
            evaluation: Some(Evaluation::Single(fitness)),
            rank: None,
            crowding_distance: None,
        }
    }

    /// Creates an individual with known multi-objective values.
    pub fn with_objectives(genes: Vec<GeneValue>, objectives: Vec<f64>) -> Self {
        Self {
            genes,
            evaluation: Some(Evaluation::Multi(objectives)),
            rank: None,
            crowding_distance: None,
        }
    }

    /// Stores a fresh evaluation and clears NSGA-II metadata.
    pub fn set_evaluation(&mut self, evaluation: Evaluation) {
        self.evaluation = Some(evaluation);
        self.rank = None;
        self.crowding_distance = None;
    }

    /// Returns fitness or panics if fitness is not available.
    pub fn fitness_or_panic(&self) -> f64 {
        self.evaluation
            .as_ref()
            .expect("individual evaluation should be available")
            .as_single()
            .expect("individual evaluation should be single-objective")
    }

    /// Returns objectives or panics if not available.
    pub fn objectives_or_panic(&self) -> &[f64] {
        self.evaluation
            .as_ref()
            .expect("individual fitness should be evaluated")
            .as_multi()
            .expect("individual evaluation should be multi-objective")
    }

    /// Clears evaluation and NSGA-II metadata after mutation/crossover changes genes.
    pub fn clear_evaluation(&mut self) {
        self.evaluation = None;
        self.rank = None;
        self.crowding_distance = None;
    }
}
