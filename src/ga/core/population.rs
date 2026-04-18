use crate::ga::core::individual::Individual;
use crate::ga::error::GaError;
/// Population container and fitness aggregation helpers.

/// Collection of individuals in one generation.
#[derive(Debug, Clone)]
pub struct Population {
    /// Individuals in the current population.
    pub individuals: Vec<Individual>,
}

impl Population {
    /// Creates a population from an individual list.
    pub fn new(individuals: Vec<Individual>) -> Self {
        Self { individuals }
    }

    /// Creates an empty population.
    pub fn empty() -> Self {
        Self {
            individuals: Vec::new(),
        }
    }

    /// Returns the number of individuals.
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Returns whether the population is empty.
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// Returns the highest-fitness individual in single-objective mode.
    pub fn best(&self) -> Result<&Individual, GaError> {
        self.individuals
            .iter()
            .max_by(|left, right| {
                left.fitness_or_panic()
                    .partial_cmp(&right.fitness_or_panic())
                    .expect("fitness comparison failed")
            })
            .ok_or(GaError::EmptyPopulation)
    }

    /// Returns the first Pareto front as cloned individuals.
    pub fn pareto_front(&self) -> Vec<Individual> {
        self.individuals
            .iter()
            .filter(|individual| individual.rank == Some(0))
            .cloned()
            .collect()
    }
}
