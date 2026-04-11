use crate::ga::analysis::stats::RunStats;
/// Stop criteria used by the evolutionary loop.

/// Termination policy for GA execution.
#[derive(Debug, Clone)]
pub enum StopCondition {
    /// Stop when current generation reaches max_generations.
    MaxGenerations,

    /// Stop when best fitness reaches or exceeds the target.
    TargetFitness(f64),

    /// Stop when the best fitness does not improve for N generations.
    NoImprovement { generations: usize },
}

/// Evaluates whether GA execution should stop at the current state.
pub fn should_stop(
    condition: &StopCondition,
    current_generation: usize,
    best_fitness: f64,
    stats: &RunStats,
    max_generations: usize,
) -> bool {
    match condition {
        StopCondition::MaxGenerations => current_generation >= max_generations,
        StopCondition::TargetFitness(target) => {
            current_generation >= max_generations || best_fitness >= *target
        }
        StopCondition::NoImprovement { generations } => {
            if current_generation >= max_generations {
                return true;
            }

            let history = &stats.best_fitness_per_generation;

            if history.len() <= *generations {
                return false;
            }

            let last = *history.last().expect("history should not be empty");

            let stagnant_slice = &history[history.len() - 1 - generations..history.len() - 1];

            stagnant_slice.iter().all(|fitness| *fitness >= last)
        }
    }
}
