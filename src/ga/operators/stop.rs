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

    /// Stop when any configured criterion is met.
    ///
    /// `max_generations` is always checked globally by `should_stop`.
    /// This variant lets you additionally combine target fitness and
    /// no-improvement criteria with OR semantics.
    Any {
        target_fitness: Option<f64>,
        no_improvement_generations: Option<usize>,
    },
}

fn reached_no_improvement_limit(stats: &RunStats, generations: usize) -> bool {
    let history = &stats.best_fitness_per_generation;

    if history.len() <= generations {
        return false;
    }

    let last = *history.last().expect("history should not be empty");
    let stagnant_slice = &history[history.len() - 1 - generations..history.len() - 1];

    stagnant_slice.iter().all(|fitness| *fitness >= last)
}

/// Evaluates whether GA execution should stop at the current state.
pub fn should_stop(
    condition: &StopCondition,
    current_generation: usize,
    best_fitness: f64,
    stats: &RunStats,
    max_generations: usize,
) -> bool {
    if current_generation >= max_generations {
        return true;
    }

    match condition {
        StopCondition::MaxGenerations => false,
        StopCondition::TargetFitness(target) => best_fitness >= *target,
        StopCondition::NoImprovement { generations } => {
            reached_no_improvement_limit(stats, *generations)
        }
        StopCondition::Any {
            target_fitness,
            no_improvement_generations,
        } => {
            let target_reached = target_fitness.is_some_and(|target| best_fitness >= target);
            let no_improvement_reached = no_improvement_generations
                .is_some_and(|generations| reached_no_improvement_limit(stats, generations));

            target_reached || no_improvement_reached
        }
    }
}
