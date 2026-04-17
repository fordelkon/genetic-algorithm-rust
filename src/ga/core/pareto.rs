use crate::ga::core::gene::GeneValue;
use crate::ga::core::individual::Individual;

/// Public view of a non-dominated solution.
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoSolution {
    /// Decision-variable genes of the solution.
    pub genes: Vec<GeneValue>,

    /// Objective values associated with the solution.
    pub objectives: Vec<f64>,

    /// Pareto rank assigned by NSGA-II.
    pub rank: usize,

    /// Crowding-distance estimate used to preserve diversity.
    pub crowding_distance: f64,
}

impl ParetoSolution {
    /// Builds a public Pareto solution snapshot from an evaluated individual.
    pub fn from_individual(individual: &Individual) -> Self {
        Self {
            genes: individual.genes.clone(),
            objectives: individual.objectives_or_panic().to_vec(),
            rank: individual.rank.unwrap_or(0),
            crowding_distance: individual.crowding_distance.unwrap_or(0.0),
        }
    }
}
