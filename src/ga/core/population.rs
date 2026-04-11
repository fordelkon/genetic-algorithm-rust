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

    /// Returns the highest-fitness individual.
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

    /// Sorts the population by fitness in descending order.
    pub fn sort_by_fitness_desc(&mut self) {
        self.individuals.sort_by(|left, right| {
            right
                .fitness_or_panic()
                .partial_cmp(&left.fitness_or_panic())
                .expect("fitness comparison failed")
        });
    }

    /// Returns clones of the top-k elite individuals.
    pub fn elite(&self, count: usize) -> Vec<Individual> {
        let mut sorted = self.individuals.clone();
        sorted.sort_by(|left, right| {
            right
                .fitness_or_panic()
                .partial_cmp(&left.fitness_or_panic())
                .expect("fitness comparison failed")
        });
        sorted.into_iter().take(count).collect()
    }

    /// Returns average fitness across all individuals.
    pub fn average_fitness(&self) -> Result<f64, GaError> {
        if self.is_empty() {
            return Err(GaError::EmptyPopulation);
        }

        let total = self
            .individuals
            .iter()
            .map(Individual::fitness_or_panic)
            .sum::<f64>();
        Ok(total / self.len() as f64)
    }

    /// Returns population standard deviation of fitness.
    pub fn fitness_std_dev(&self) -> Result<f64, GaError> {
        let mean = self.average_fitness()?;
        let variance = self
            .individuals
            .iter()
            .map(|individual| {
                let delta = individual.fitness_or_panic() - mean;
                delta * delta
            })
            .sum::<f64>()
            / self.len() as f64;
        Ok(variance.sqrt())
    }
}
