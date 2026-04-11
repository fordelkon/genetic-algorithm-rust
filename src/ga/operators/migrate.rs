/// Migration topology helpers for island-model evolution.

/// Migration topology used to route emigrants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationType {
    /// Each island sends migrants to one neighbor in a ring.
    Ring,
    /// Each island sends migrants to all other islands.
    FullyConnected,
}

/// Returns destination island indices for a source island under a topology.
pub fn migrate(
    migration_type: &MigrationType,
    island_index: usize,
    num_islands: usize,
) -> Vec<usize> {
    match migration_type {
        MigrationType::Ring => vec![(island_index + 1) % num_islands],
        MigrationType::FullyConnected => (0..num_islands).filter(|&i| i != island_index).collect(),
    }
}
