use crate::ga::{
    core::gene::{GeneScalarType, GeneValue, GenesDomain, GenesValueType},
    error::GaError,
    operators::crossover::CrossoverType,
    operators::mutation::MutationType,
    operators::selection::SelectionType,
    operators::stop::StopCondition,
};

pub use crate::ga::operators::migrate::MigrationType;
/// Engine configuration and builder APIs.

/// Immutable runtime configuration for one GA run.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    // ==========================================
    // Basic parameters
    pub population_size: usize,

    pub num_genes: usize,

    pub num_generations: usize,

    // ==========================================
    // Initialization and domain
    pub init_range_low: f64,

    pub init_range_high: f64,

    pub genes_value_type: GenesValueType,

    pub genes_domain: Option<GenesDomain>,

    // ==========================================
    // Selection
    pub num_parents_mating: usize,

    pub selection_type: SelectionType,

    pub elitism_count: usize,

    // ==========================================
    // Crossover
    pub crossover_type: CrossoverType,

    pub crossover_probability: f64,

    // ==========================================
    // Mutation
    pub mutation_type: MutationType,

    pub mutation_probability: f64,

    pub mutation_num_genes: Option<usize>,

    // ==========================================
    // Control and reproducibility
    pub random_seed: Option<u64>,

    pub stop_condition: StopCondition,

    // ==========================================
    // Island-model settings
    pub num_islands: usize,

    pub migration_count: usize,

    pub migration_interval: usize,

    pub migration_topology: MigrationType,
}

/// Builder for [`EngineConfig`].
/// ```rust
/// # use genetic_algorithm_rust::{CrossoverType, EngineConfig, MutationType};
/// let config = EngineConfig::builder(100, 10, 50, 50)
///     .crossover(CrossoverType::TwoPoint, 0.85)
///     .mutation_probability(0.1)
///     .init_range(-10.0, 10.0)
///     .build()
///     .expect("valid config");
/// ```
#[derive(Debug, Clone)]
pub struct EngineConfigBuilder {
    config: EngineConfig,
}

impl EngineConfig {
    /// Creates a new [`EngineConfigBuilder`].
    pub fn builder(
        population_size: usize,
        num_genes: usize,
        num_generations: usize,
        num_parents_mating: usize,
    ) -> EngineConfigBuilder {
        EngineConfigBuilder::new(
            population_size,
            num_genes,
            num_generations,
            num_parents_mating,
        )
    }

    /// Validates all configuration constraints.
    pub(crate) fn validate(&self) -> Result<(), GaError> {
        if self.population_size == 0 {
            return Err(GaError::InvalidConfig(
                "population_size must be greater than 0".into(),
            ));
        }

        if self.num_genes == 0 {
            return Err(GaError::InvalidConfig(
                "num_genes must be greater than 0".into(),
            ));
        }

        if self.num_generations == 0 {
            return Err(GaError::InvalidConfig(
                "num_generations must be greater than 0".into(),
            ));
        }

        if self.num_parents_mating < 2 || self.num_parents_mating > self.population_size {
            return Err(GaError::InvalidConfig(
                "num_parents_mating must be between 2 and population_size".into(),
            ));
        }

        if self.elitism_count > self.population_size {
            return Err(GaError::InvalidConfig(
                "elitism_count must not exceed population_size".into(),
            ));
        }

        if !(0.0..=1.0).contains(&self.crossover_probability) {
            return Err(GaError::InvalidConfig(
                "crossover_probability must be between 0 and 1".into(),
            ));
        }

        if !(0.0..=1.0).contains(&self.mutation_probability) {
            return Err(GaError::InvalidConfig(
                "mutation_probability must be between 0 and 1".into(),
            ));
        }

        if let Some(mutation_num_genes) = self.mutation_num_genes
            && mutation_num_genes > self.num_genes
        {
            return Err(GaError::InvalidConfig(
                "mutation_num_genes must not exceed num_genes".into(),
            ));
        }

        match &self.genes_value_type {
            GenesValueType::PerGene(types) if types.len() != self.num_genes => {
                return Err(GaError::InvalidConfig(
                    "per-gene genes_value_type length must match num_genes".into(),
                ));
            }
            GenesValueType::All(scalar_type) if !scalar_type.is_supported() => {
                return Err(GaError::UnsupportedGeneType(scalar_type.as_str().into()));
            }
            GenesValueType::PerGene(types) => {
                for scalar_type in types {
                    if !scalar_type.is_supported() {
                        return Err(GaError::UnsupportedGeneType(scalar_type.as_str().into()));
                    }
                }
            }
            _ => {}
        }

        match &self.genes_domain {
            Some(GenesDomain::PerGene(domains)) if domains.len() != self.num_genes => {
                return Err(GaError::InvalidConfig(
                    "per-gene genes_domain length must match num_genes".into(),
                ));
            }
            Some(space) => {
                for gene_index in 0..self.num_genes {
                    let domain = space.domain_for(gene_index);
                    if let Err(message) = domain.validate() {
                        return Err(GaError::InvalidConfig(message));
                    }

                    domain.validate_for_type(self.gene_value_type_for(gene_index))?;
                }
            }
            None => {}
        }

        for gene_index in 0..self.num_genes {
            let scalar_type = self.gene_value_type_for(gene_index);
            if scalar_type.is_unsigned() && self.init_range_low < 0.0 && self.genes_domain.is_none()
            {
                return Err(GaError::InvalidConfig(format!(
                    "init_range_low {} cannot be represented by {} without genes_domain",
                    self.init_range_low,
                    scalar_type.as_str()
                )));
            }
        }

        match &self.selection_type {
            SelectionType::Tournament { k } if *k == 0 => {
                return Err(GaError::InvalidConfig(
                    "tournament k must be greater than 0".into(),
                ));
            }
            _ => {}
        }

        match &self.mutation_type {
            MutationType::AdaptiveRandomReset {
                low_quality_num_genes,
                high_quality_num_genes,
                ..
            }
            | MutationType::AdaptiveRandomPerturbation {
                low_quality_num_genes,
                high_quality_num_genes,
                ..
            } => {
                if *low_quality_num_genes > self.num_genes
                    || *high_quality_num_genes > self.num_genes
                {
                    return Err(GaError::InvalidConfig(
                        "adaptive mutation gene counts must not exceed num_genes".into(),
                    ));
                }
            }
            _ => {}
        }

        if self.num_islands == 0 {
            return Err(GaError::InvalidConfig(
                "num_islands must be at least 1".into(),
            ));
        }

        if self.num_islands > 1 {
            self.validate_island_fields()?;
            if self.migration_count >= self.population_size {
                return Err(GaError::InvalidConfig(
                    "migration_count must be less than population_size".into(),
                ));
            }
        }

        Ok(())
    }

    /// Validates island-model-only fields.
    pub(crate) fn validate_island_fields(&self) -> Result<(), GaError> {
        if self.num_islands < 2 {
            return Err(GaError::InvalidConfig(
                "island model requires at least 2 islands".into(),
            ));
        }
        if self.migration_count == 0 {
            return Err(GaError::InvalidConfig(
                "migration_count must be at least 1".into(),
            ));
        }
        if self.migration_interval == 0 {
            return Err(GaError::InvalidConfig(
                "migration_interval must be at least 1".into(),
            ));
        }

        Ok(())
    }

    /// Returns value type for a specific gene index.
    pub fn gene_value_type_for(&self, gene_index: usize) -> GeneScalarType {
        self.genes_value_type.value_type_for(gene_index)
    }

    /// Samples one gene value for initialization.
    pub fn sample_gene(&self, gene_index: usize, rng: &mut impl rand::Rng) -> GeneValue {
        let sampled = match &self.genes_domain {
            Some(space) => space.domain_for(gene_index).sample_numeric(rng),
            None => rng.gen_range(self.init_range_low..=self.init_range_high),
        };

        self.normalize_gene(gene_index, sampled)
            .expect("gene sampling should always produce a valid value")
    }

    /// Normalizes and casts a raw numeric value into a valid gene value.
    pub fn normalize_gene(&self, gene_index: usize, value: f64) -> Result<GeneValue, GaError> {
        let scalar_type = self.gene_value_type_for(gene_index);
        let typed = match &self.genes_domain {
            Some(space) => space.domain_for(gene_index).normalize_numeric(value),
            None => value,
        };

        GeneValue::cast_from_f64(scalar_type, typed)
    }
}

impl EngineConfigBuilder {
    /// Creates a builder with sensible defaults.
    pub fn new(
        population_size: usize,
        num_genes: usize,
        num_generations: usize,
        num_parents_mating: usize,
    ) -> Self {
        Self {
            config: EngineConfig {
                population_size,
                num_genes,
                num_generations,
                num_parents_mating,
                init_range_low: -4.0,
                init_range_high: 4.0,
                genes_value_type: GenesValueType::All(GeneScalarType::F64),
                genes_domain: None,
                selection_type: SelectionType::Tournament { k: 3 },
                elitism_count: 1,
                crossover_type: CrossoverType::SinglePoint,
                crossover_probability: 0.8,
                mutation_type: MutationType::RandomPerturbation {
                    min_delta: -1.0,
                    max_delta: 1.0,
                },
                mutation_probability: 0.05,
                mutation_num_genes: None,
                random_seed: None,
                stop_condition: StopCondition::MaxGenerations,
                num_islands: 1,
                migration_count: 2,
                migration_interval: 10,
                migration_topology: MigrationType::Ring,
            },
        }
    }

    /// Enables island-model settings.
    pub fn island_model(
        mut self,
        num_islands: usize,
        migration_count: usize,
        migration_interval: usize,
        migration_topology: MigrationType,
    ) -> Self {
        self.config.num_islands = num_islands;
        self.config.migration_count = migration_count;
        self.config.migration_interval = migration_interval;
        self.config.migration_topology = migration_topology;
        self
    }

    /// Sets initialization range used when no domain is provided.
    pub fn init_range(mut self, low: f64, high: f64) -> Self {
        self.config.init_range_low = low;
        self.config.init_range_high = high;
        self
    }

    /// Sets chromosome value-type schema.
    pub fn genes_value_type(mut self, genes_value_type: GenesValueType) -> Self {
        self.config.genes_value_type = genes_value_type;
        self
    }

    /// Sets optional gene domains.
    pub fn genes_domain(mut self, genes_domain: Option<GenesDomain>) -> Self {
        self.config.genes_domain = genes_domain;
        self
    }

    /// Sets parent selection strategy.
    pub fn selection_type(mut self, selection_type: SelectionType) -> Self {
        self.config.selection_type = selection_type;
        self
    }

    /// Sets elitism count.
    pub fn elitism_count(mut self, elitism_count: usize) -> Self {
        self.config.elitism_count = elitism_count;
        self
    }

    /// Sets crossover strategy.
    pub fn crossover_type(mut self, crossover_type: CrossoverType) -> Self {
        self.config.crossover_type = crossover_type;
        self
    }

    /// Sets crossover probability.
    pub fn crossover_probability(mut self, crossover_probability: f64) -> Self {
        self.config.crossover_probability = crossover_probability;
        self
    }

    /// Sets crossover strategy and probability together.
    pub fn crossover(mut self, crossover_type: CrossoverType, crossover_probability: f64) -> Self {
        self.config.crossover_type = crossover_type;
        self.config.crossover_probability = crossover_probability;
        self
    }

    /// Sets mutation strategy.
    pub fn mutation_type(mut self, mutation_type: MutationType) -> Self {
        self.config.mutation_type = mutation_type;
        self
    }

    /// Sets mutation probability.
    pub fn mutation_probability(mut self, mutation_probability: f64) -> Self {
        self.config.mutation_probability = mutation_probability;
        self
    }

    /// Sets a fixed number of genes to mutate per individual.
    pub fn mutation_num_genes(mut self, mutation_num_genes: usize) -> Self {
        self.config.mutation_num_genes = Some(mutation_num_genes);
        self
    }

    /// Sets mutation strategy and probability together.
    pub fn mutation(mut self, mutation_type: MutationType, mutation_probability: f64) -> Self {
        self.config.mutation_type = mutation_type;
        self.config.mutation_probability = mutation_probability;
        self
    }

    /// Sets optional random seed for reproducibility.
    pub fn random_seed(mut self, random_seed: Option<u64>) -> Self {
        self.config.random_seed = random_seed;
        self
    }

    /// Sets stop condition.
    pub fn stop_condition(mut self, stop_condition: StopCondition) -> Self {
        self.config.stop_condition = stop_condition;
        self
    }

    /// Validates and builds final configuration.
    pub fn build(self) -> Result<EngineConfig, GaError> {
        self.config.validate()?;
        Ok(self.config)
    }
}
