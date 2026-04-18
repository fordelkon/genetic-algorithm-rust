pub mod ga;

pub use ga::analysis::report::ExperimentSummary;
pub use ga::analysis::report::ParetoExperimentSummary;
pub use ga::analysis::visualize::VisualizationOptions;
pub use ga::core::evaluation::Evaluation;
pub use ga::core::gene::{GeneDomain, GeneScalarType, GeneValue, GenesDomain, GenesValueType};
pub use ga::core::pareto::ParetoSolution;
pub use ga::engine::config::{EngineConfig, EngineConfigBuilder, MigrationType, OptimizationMode};
pub use ga::engine::engine::{EngineKernel, EngineRunResult, EvolutionEngine, IslandEngine};
pub use ga::error::GaError;
pub use ga::operators::{
    crossover::CrossoverType, mutation::MutationType, selection::SelectionType, stop::StopCondition,
};
