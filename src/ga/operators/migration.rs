/// Migration topology helpers for island-model evolution.

use crate::ga::core::individual::Individual;
use crate::ga::engine::config::OptimizationMode;
use crate::ga::engine::engine::EngineKernel;

/// Migration topology used to route emigrants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationType {
    /// Each island sends migrants to one neighbor in a ring.
    Ring,
    /// Each island sends migrants to all other islands.
    FullyConnected,
}

/// Returns destination island indices for a source island under a topology.
pub fn destinations(
    migration_type: &MigrationType,
    island_index: usize,
    num_islands: usize,
) -> Vec<usize> {
    match migration_type {
        MigrationType::Ring => vec![(island_index + 1) % num_islands],
        MigrationType::FullyConnected => (0..num_islands).filter(|&i| i != island_index).collect(),
    }
}

/// Migrates leading individuals between islands according to the configured topology.
pub fn migrate(
    islands: &mut [EngineKernel],
    migration_type: &MigrationType,
    migration_count: usize,
    optimization_mode: &OptimizationMode,
) {
    let k = migration_count;
    let n = islands.len();

    let emigrants: Vec<Vec<Individual>> = islands
        .iter()
        .map(|island| match optimization_mode {
            OptimizationMode::SingleObjective => island.population.elite(k),
            OptimizationMode::Nsga2 { .. } => island
                .population
                .sorted_nsga2()
                .expect("NSGA-II migration requires ranking metadata")
                .into_iter()
                .take(k)
                .collect(),
        })
        .collect();

    for (src, src_emigrants) in emigrants.iter().enumerate() {
        let neighbors = destinations(migration_type, src, n);
        for &dst in &neighbors {
            let island = &mut islands[dst];
            match optimization_mode {
                OptimizationMode::SingleObjective => island.population.sort_by_fitness_desc(),
                OptimizationMode::Nsga2 { .. } => {
                    island.population.individuals = island
                        .population
                        .sorted_nsga2()
                        .expect("NSGA-II migration requires ranking metadata");
                }
            }

            let pop_len = island.population.individuals.len();
            let replace_start = pop_len.saturating_sub(k);
            for (j, emigrant) in src_emigrants.iter().enumerate() {
                if replace_start + j < pop_len {
                    island.population.individuals[replace_start + j] = emigrant.clone();
                }
            }

            if matches!(optimization_mode, OptimizationMode::Nsga2 { .. }) {
                island
                    .population
                    .assign_nsga2_metadata()
                    .expect("NSGA-II migration re-ranking should succeed");
            }
        }
    }
}
