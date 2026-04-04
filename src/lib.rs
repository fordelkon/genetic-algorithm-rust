pub mod ga;

pub use ga::config::{GaConfig, GaConfigBuilder};
pub use ga::crossover::CrossoverType;
pub use ga::engine::GeneticAlgorithm;
pub use ga::error::GaError;
pub use ga::gene::{GeneDomain, GeneScalarType, GeneValue, GenesDomain, GenesValueType};
pub use ga::mutation::MutationType;
pub use ga::report::ExperimentSummary;
pub use ga::selection::SelectionType;
pub use ga::stop::StopCondition;
pub use ga::visualize::VisualizationOptions;
