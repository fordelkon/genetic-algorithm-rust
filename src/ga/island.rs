use crate::ga::{
    config::GaConfig, engine::GeneticAlgorithm, error::GaError, gene::GeneValue,
    individual::Individual, stats::RunStats,
};

/// Migration topology defining which islands exchange individuals.
///
/// The topology determines the neighbor structure: during migration, each island
/// sends its best individuals to its neighbors and receives migrants from them.
#[derive(Debug, Clone)]
pub enum MigrationTopology {
    /// Each island sends migrants to the next island in a circular arrangement.
    /// Island `i` sends to island `(i + 1) % n`.
    Ring,
    /// Every island sends migrants to every other island.
    FullyConnected,
}

impl MigrationTopology {
    /// Returns the list of destination island indices for a given source island.
    pub fn neighbors(&self, island_index: usize, num_islands: usize) -> Vec<usize> {
        match self {
            Self::Ring => vec![(island_index + 1) % num_islands],
            Self::FullyConnected => (0..num_islands)
                .filter(|&i| i != island_index)
                .collect(),
        }
    }
}

/// Configuration for the island model.
///
/// Controls how many islands run in parallel, how often migration occurs,
/// how many individuals migrate, and the topology connecting islands.
#[derive(Debug, Clone)]
pub struct IslandConfig {
    /// Number of islands (sub-populations).
    pub num_islands: usize,
    /// Number of best individuals to migrate from each island per migration event.
    pub migration_count: usize,
    /// Migration occurs every `migration_interval` generations.
    pub migration_interval: usize,
    /// The topology defining neighbor relationships between islands.
    pub topology: MigrationTopology,
}

impl IslandConfig {
    /// Create a new island configuration.
    ///
    /// # Arguments
    ///
    /// * `num_islands` - Number of sub-populations (must be >= 2).
    /// * `migration_count` - Number of elite individuals to migrate per event.
    /// * `migration_interval` - Generations between migration events.
    /// * `topology` - The migration topology.
    pub fn new(
        num_islands: usize,
        migration_count: usize,
        migration_interval: usize,
        topology: MigrationTopology,
    ) -> Self {
        Self {
            num_islands,
            migration_count,
            migration_interval,
            topology,
        }
    }

    fn validate(&self) -> Result<(), GaError> {
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
}

/// Island model genetic algorithm.
///
/// Runs multiple independent `GeneticAlgorithm` instances (islands) with periodic
/// migration of elite individuals between them according to a `MigrationTopology`.
///
/// Each island evolves its own population independently. Every `migration_interval`
/// generations, the best individuals from each island are copied to neighboring
/// islands (replacing the worst individuals there). This maintains population
/// diversity while allowing good solutions to spread across islands.
///
/// # Type Parameters
///
/// * `F` - The fitness function type. Must be `Clone` so each island gets its own copy.
pub struct IslandModel<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync + Clone,
{
    /// The island configuration.
    pub island_config: IslandConfig,
    /// The individual GA instances (one per island).
    pub islands: Vec<GeneticAlgorithm<F>>,
}

impl<F> IslandModel<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync + Clone,
{
    /// Create a new island model where all islands share the same GA configuration.
    ///
    /// Each island gets a deterministic RNG seed derived from the base config's
    /// `random_seed` (if set) combined with the island index, ensuring
    /// reproducible but independent evolution across islands.
    ///
    /// # Errors
    ///
    /// Returns `GaError::InvalidConfig` if the island or GA configuration is invalid,
    /// or if `migration_count >= population_size`.
    pub fn new(
        island_config: IslandConfig,
        ga_config: GaConfig,
        fitness_fn: F,
    ) -> Result<Self, GaError> {
        island_config.validate()?;
        if island_config.migration_count >= ga_config.population_size {
            return Err(GaError::InvalidConfig(
                "migration_count must be less than population_size".into(),
            ));
        }

        let mut islands = Vec::with_capacity(island_config.num_islands);
        for i in 0..island_config.num_islands {
            let mut config = ga_config.clone();
            // Derive a unique seed per island for reproducibility
            config.random_seed = ga_config.random_seed.map(|seed| seed.wrapping_add(i as u64));
            islands.push(GeneticAlgorithm::new(config, fitness_fn.clone())?);
        }

        Ok(Self {
            island_config,
            islands,
        })
    }

    /// Create a new island model with different GA configurations per island.
    ///
    /// This allows heterogeneous islands with different selection strategies,
    /// mutation rates, etc. — a powerful technique for maintaining diversity.
    ///
    /// # Errors
    ///
    /// Returns `GaError::InvalidConfig` if the number of configs doesn't match
    /// `num_islands`, or if any individual config is invalid.
    pub fn with_configs(
        island_config: IslandConfig,
        ga_configs: Vec<GaConfig>,
        fitness_fn: F,
    ) -> Result<Self, GaError> {
        island_config.validate()?;
        if ga_configs.len() != island_config.num_islands {
            return Err(GaError::InvalidConfig(format!(
                "expected {} GA configs, got {}",
                island_config.num_islands,
                ga_configs.len()
            )));
        }
        for config in &ga_configs {
            if island_config.migration_count >= config.population_size {
                return Err(GaError::InvalidConfig(
                    "migration_count must be less than population_size".into(),
                ));
            }
        }

        let mut islands = Vec::with_capacity(island_config.num_islands);
        for (i, mut config) in ga_configs.into_iter().enumerate() {
            config.random_seed = config.random_seed.map(|seed| seed.wrapping_add(i as u64));
            islands.push(GeneticAlgorithm::new(config, fitness_fn.clone())?);
        }

        Ok(Self {
            island_config,
            islands,
        })
    }

    /// Perform one migration event: copy elite individuals between islands.
    ///
    /// For each island, the top `migration_count` individuals are cloned and
    /// sent to all neighbor islands (determined by the topology). At each
    /// receiving island, the worst individuals are replaced by the incoming
    /// migrants.
    fn migrate(&mut self) {
        let k = self.island_config.migration_count;
        let n = self.islands.len();

        // Collect emigrants from each island (cloned elite individuals)
        let emigrants: Vec<Vec<Individual>> = self
            .islands
            .iter()
            .map(|island| island.population.elite(k))
            .collect();

        // Deliver emigrants to neighbors
        for (src, src_emigrants) in emigrants.iter().enumerate() {
            let neighbors = self.island_config.topology.neighbors(src, n);
            for &dst in &neighbors {
                let island = &mut self.islands[dst];
                // Sort so worst are at the end, then replace them
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

    /// Run the island model to completion.
    ///
    /// 1. Initialize and evaluate all island populations.
    /// 2. Evolve each island independently for `migration_interval` generations.
    /// 3. Migrate elite individuals between islands.
    /// 4. Repeat until any island reaches its `num_generations` limit.
    ///
    /// Returns per-island statistics.
    ///
    /// # Errors
    ///
    /// Returns `GaError` if any island encounters an error during evolution.
    pub fn run(&mut self) -> Result<Vec<&RunStats>, GaError> {
        // Initialize all islands
        for island in &mut self.islands {
            island.initialize_population()?;
            island.evaluate_population();
            island.stats.record(&island.population)?;
        }

        let max_generations = self
            .islands
            .iter()
            .map(|island| island.config.num_generations)
            .max()
            .unwrap_or(0);

        let mut global_generation = 0;

        while global_generation < max_generations {
            // Run each island for one migration interval (or until done)
            let steps = self
                .island_config
                .migration_interval
                .min(max_generations - global_generation);

            for _ in 0..steps {
                for island in &mut self.islands {
                    if island.generation < island.config.num_generations {
                        island.next_generation()?;
                        island.generation += 1;
                    }
                }
                global_generation += 1;
            }

            // Migrate if we haven't reached the end
            if global_generation < max_generations {
                self.migrate();
            }
        }

        Ok(self.islands.iter().map(|island| &island.stats).collect())
    }

    /// Returns the best individual found across all islands.
    ///
    /// # Errors
    ///
    /// Returns `GaError::EmptyPopulation` if all islands are empty.
    pub fn best_solution(&self) -> Result<&Individual, GaError> {
        self.islands
            .iter()
            .filter_map(|island| island.best_solution().ok())
            .max_by(|a, b| {
                a.fitness_or_panic()
                    .partial_cmp(&b.fitness_or_panic())
                    .expect("fitness comparison failed")
            })
            .ok_or(GaError::EmptyPopulation)
    }

    /// Returns the global best fitness found across all islands and all generations.
    pub fn best_fitness(&self) -> Option<f64> {
        self.islands
            .iter()
            .filter_map(|island| island.stats.best_fitness)
            .max_by(|a, b| a.partial_cmp(b).expect("fitness comparison failed"))
    }
}
