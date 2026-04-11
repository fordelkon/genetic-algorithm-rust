pub mod ga;

pub use ga::analysis::report::ExperimentSummary;
pub use ga::analysis::visualize::VisualizationOptions;
pub use ga::core::gene::{GeneDomain, GeneScalarType, GeneValue, GenesDomain, GenesValueType};
pub use ga::engine::config::{
    EngineExecutionMode, GaConfig, GaConfigBuilder, IslandConfig, MigrationTopology,
};
pub use ga::engine::engine::{EngineRunResult, EvolutionEngine, GeneticAlgorithm, IslandModel};
pub use ga::error::GaError;
pub use ga::operators::{
    crossover::CrossoverType, mutation::MutationType, selection::SelectionType, stop::StopCondition,
};
