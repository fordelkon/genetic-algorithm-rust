use crate::ga::core::gene::GeneValue;
/// Individual representation for GA evolution.

/// A candidate solution with genes and optional fitness.
#[derive(Debug, Clone, PartialEq)]
pub struct Individual {
    /// Chromosome values.
    pub genes: Vec<GeneValue>,
    /// Fitness score, present after evaluation.
    pub fitness: Option<f64>,
}

impl Individual {
    /// Creates a new unevaluated individual.
    pub fn new(genes: Vec<GeneValue>) -> Self {
        Self {
            genes,
            fitness: None,
        }
    }

    /// Creates an individual with a known fitness value.
    pub fn with_fitness(genes: Vec<GeneValue>, fitness: f64) -> Self {
        Self {
            genes,
            fitness: Some(fitness),
        }
    }

    /// Returns fitness or panics if fitness is not available.
    pub fn fitness_or_panic(&self) -> f64 {
        self.fitness
            .expect("individual fitness should be evaluated")
    }
}
