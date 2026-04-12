/// Migration topology helpers for island-model evolution.

use crate::ga::core::{gene::GeneValue, individual::Individual};
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

/// Migrates top individuals between islands according to the configured topology.
pub fn migrate<F>(
    islands: &mut [EngineKernel<F>],
    migration_type: &MigrationType,
    migration_count: usize,
) where
    F: Fn(&[GeneValue]) -> f64 + Sync,
{
    let k = migration_count;
    let n = islands.len();

    let emigrants: Vec<Vec<Individual>> = islands
        .iter()
        .map(|island| island.population.elite(k))
        .collect();

    for (src, src_emigrants) in emigrants.iter().enumerate() {
        let neighbors = destinations(migration_type, src, n);
        for &dst in &neighbors {
            let island = &mut islands[dst];
            island.population.sort_by_fitness_desc();
            let pop_len = island.population.individuals.len();
            let replace_start = pop_len.saturating_sub(k);
            for (j, emigrant) in src_emigrants.iter().enumerate() {
                if replace_start + j < pop_len {
                    island.population.individuals[replace_start + j] = emigrant.clone();
                }
            }
        }
    }
}
