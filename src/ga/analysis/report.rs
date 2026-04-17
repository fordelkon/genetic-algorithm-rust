use crate::ga::analysis::stats::RunStats;
use crate::ga::error::GaError;
/// Experiment-level summary extraction from run statistics.

/// Compact summary metrics for a single-objective GA run.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSummary {
    pub generations: usize,
    pub best_fitness: f64,
    pub final_avg_fitness: f64,
    pub final_std_fitness: f64,
    pub initial_best_fitness: f64,
    pub final_best_fitness: f64,
    pub improvement: f64,
    pub best_genes: Vec<f64>,
}

/// Compact summary metrics for an NSGA-II run.
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoExperimentSummary {
    pub generations: usize,
    pub final_front_size: usize,
    pub final_front_count: usize,
}

impl ExperimentSummary {
    /// Builds a summary from recorded run statistics.
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

impl ParetoExperimentSummary {
    pub fn from_stats(stats: &RunStats) -> Result<Self, GaError> {
        let multi = stats.multi_objective.as_ref().ok_or_else(|| {
            GaError::Visualization("multi-objective history is unavailable".into())
        })?;

        let generations = multi.front_0_size_per_generation.len();
        if generations == 0 {
            return Err(GaError::Visualization(
                "cannot summarize an empty NSGA-II run history".into(),
            ));
        }

        Ok(Self {
            generations,
            final_front_size: *multi
                .front_0_size_per_generation
                .last()
                .expect("non-empty history should have last item"),
            final_front_count: *multi
                .front_count_per_generation
                .last()
                .expect("non-empty history should have last item"),
        })
    }
}
