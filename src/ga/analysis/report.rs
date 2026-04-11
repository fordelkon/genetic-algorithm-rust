use crate::ga::analysis::stats::RunStats;
use crate::ga::error::GaError;
/// Experiment-level summary extraction from run statistics.

/// Compact summary metrics for a single GA run.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSummary {
    /// Number of recorded generations.
    pub generations: usize,
    /// Global best fitness across the run.
    pub best_fitness: f64,
    /// Average fitness of the final generation.
    pub final_avg_fitness: f64,
    /// Fitness standard deviation of the final generation.
    pub final_std_fitness: f64,
    /// Best fitness at generation zero.
    pub initial_best_fitness: f64,
    /// Best fitness at the last generation.
    pub final_best_fitness: f64,
    /// Improvement from initial to final best fitness.
    pub improvement: f64,
    /// Best solution genes converted to f64 values.
    pub best_genes: Vec<f64>,
}

impl ExperimentSummary {
    /// Builds a summary from recorded run statistics.
    /// # Errors
    pub fn from_stats(stats: &RunStats) -> Result<Self, GaError> {
        let generations = stats.best_fitness_per_generation.len();
        if generations == 0 {
            return Err(GaError::Visualization(
                "cannot summarize an empty run history".into(),
            ));
        }

        let initial_best_fitness = stats.best_fitness_per_generation[0];
        let final_best_fitness = *stats
            .best_fitness_per_generation
            .last()
            .expect("non-empty history should have last item");
        let final_avg_fitness = *stats
            .avg_fitness_per_generation
            .last()
            .ok_or_else(|| GaError::Visualization("average fitness history is empty".into()))?;
        let best_fitness = stats
            .best_fitness
            .ok_or_else(|| GaError::Visualization("global best fitness is missing".into()))?;
        let final_std_fitness = *stats
            .std_fitness_per_generation
            .last()
            .ok_or_else(|| GaError::Visualization("fitness std-dev history is empty".into()))?;
        let best_genes = stats
            .best_solution
            .as_ref()
            .ok_or_else(|| GaError::Visualization("global best solution is missing".into()))?
            .iter()
            .map(|gene| gene.to_f64())
            .collect();

        Ok(Self {
            generations,
            best_fitness,
            final_avg_fitness,
            final_std_fitness,
            initial_best_fitness,
            final_best_fitness,
            improvement: final_best_fitness - initial_best_fitness,
            best_genes,
        })
    }
}
