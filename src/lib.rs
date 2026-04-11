pub mod ga;

pub use ga::analysis::report::ExperimentSummary;
pub use ga::analysis::visualize::VisualizationOptions;
pub use ga::core::gene::{GeneDomain, GeneScalarType, GeneValue, GenesDomain, GenesValueType};
pub use ga::engine::config::{EngineConfig, EngineConfigBuilder, MigrationType};
pub use ga::engine::engine::{EngineKernel, EngineRunResult, EvolutionEngine, IslandModel};
pub use ga::error::GaError;
pub use ga::operators::{
    crossover::CrossoverType, mutation::MutationType, selection::SelectionType, stop::StopCondition,
};
