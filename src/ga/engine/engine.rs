use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::ga::analysis::stats::RunStats;
use crate::ga::core::{gene::GeneValue, individual::Individual, population::Population};
use crate::ga::engine::config::{EngineExecutionMode, GaConfig};
use crate::ga::error::GaError;
use crate::ga::{operators::crossover, operators::mutation, operators::selection, operators::stop};

pub use crate::ga::engine::config::{IslandConfig, MigrationTopology};

/// 单群体演化的统一执行内核。
pub struct EngineKernel<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync,
{
    pub config: GaConfig,
    pub fitness_fn: F,
    pub population: Population,
    pub generation: usize,
    pub stats: RunStats,
    rng: StdRng,
}

impl<F> EngineKernel<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync,
{
    pub fn new(config: GaConfig, fitness_fn: F) -> Result<Self, GaError> {
        config.validate()?;
        let rng = Self::build_rng(config.random_seed);

        Ok(Self {
            config,
            fitness_fn,
            population: Population::empty(),
            generation: 0,
            stats: RunStats::default(),
            rng,
        })
    }

    pub fn build_rng(seed: Option<u64>) -> StdRng {
        match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => StdRng::seed_from_u64(rand::random()),
        }
    }

    pub fn initialize_population(&mut self) -> Result<(), GaError> {
        let mut individuals = Vec::with_capacity(self.config.population_size);

        for _ in 0..self.config.population_size {
            let genes = (0..self.config.num_genes)
                .map(|gene_index| self.config.sample_gene(gene_index, &mut self.rng))
                .collect::<Vec<_>>();
            individuals.push(Individual::new(genes));
        }

        self.population = Population::new(individuals);
        Ok(())
    }

    pub fn evaluate_population(&mut self) {
        self.population
            .individuals
            .par_iter_mut()
            .for_each(|individual| {
                let fitness = (self.fitness_fn)(&individual.genes);
                individual.fitness = Some(fitness);
            });
    }

    pub fn select_parents(&mut self) -> Vec<Individual> {
        selection::select_parents(
            &self.population,
            &self.config.selection_type,
            self.config.num_parents_mating,
            &mut self.rng,
        )
    }

    pub fn crossover(&mut self, parents: &[Individual]) -> Vec<Individual> {
        let offspring_count = self
            .config
            .population_size
            .saturating_sub(self.config.elitism_count);
        crossover::crossover(
            parents,
            &self.config.crossover_type,
            self.config.crossover_probability,
            offspring_count,
            &mut self.rng,
        )
    }

    pub fn mutate(&mut self, offspring: &mut [Individual]) {
        if matches!(
            self.config.mutation_type,
            mutation::MutationType::AdaptiveRandomReset { .. }
                | mutation::MutationType::AdaptiveRandomPerturbation { .. }
        ) {
            offspring.par_iter_mut().for_each(|individual| {
                let fitness = (self.fitness_fn)(&individual.genes);
                individual.fitness = Some(fitness);
            });
        }

        mutation::mutate(
            offspring,
            &self.config,
            &self.config.mutation_type,
            self.config.mutation_probability,
            &mut self.rng,
        );
    }

    pub fn next_generation(&mut self) -> Result<(), GaError> {
        let parents = self.select_parents();
        let mut offspring = self.crossover(&parents);
        self.mutate(&mut offspring);

        let mut next_individuals = self.population.elite(self.config.elitism_count);
        next_individuals.extend(offspring);

        if next_individuals.len() > self.config.population_size {
            next_individuals.truncate(self.config.population_size);
        }

        self.population = Population::new(next_individuals);
        self.evaluate_population();
        self.stats.record(&self.population)?;
        Ok(())
    }

    pub fn run(&mut self) -> Result<&RunStats, GaError> {
        self.initialize_population()?;
        self.evaluate_population();
        self.stats.record(&self.population)?;

        while !stop::should_stop(
            &self.config.stop_condition,
            self.generation,
            self.best_solution()?.fitness_or_panic(),
            &self.stats,
            self.config.num_generations,
        ) {
            self.next_generation()?;
            self.generation += 1;
        }

        Ok(&self.stats)
    }

    pub fn best_solution(&self) -> Result<&Individual, GaError> {
        self.population.best()
    }
}

/// 兼容别名：单群体 GA 使用统一内核实现。
pub type GeneticAlgorithm<F> = EngineKernel<F>;

/// Island model genetic algorithm.
pub struct IslandModel<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync + Clone,
{
    pub island_config: IslandConfig,
    pub islands: Vec<GeneticAlgorithm<F>>,
}

impl<F> IslandModel<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync + Clone,
{
    fn validate_uniform_island_settings(
        reference: &IslandConfig,
        config: &GaConfig,
        index: usize,
    ) -> Result<(), GaError> {
        if config.execution_mode != EngineExecutionMode::IslandModel {
            return Err(GaError::InvalidConfig(format!(
                "ga_configs[{index}] must use EngineExecutionMode::IslandModel"
            )));
        }

        let island = config.island_config();
        if island.num_islands != reference.num_islands
            || island.migration_count != reference.migration_count
            || island.migration_interval != reference.migration_interval
            || island.topology != reference.topology
        {
            return Err(GaError::InvalidConfig(format!(
                "ga_configs[{index}] island settings must match the shared island configuration"
            )));
        }

        Ok(())
    }

    pub fn new(ga_config: GaConfig, fitness_fn: F) -> Result<Self, GaError> {
        if ga_config.execution_mode != EngineExecutionMode::IslandModel {
            return Err(GaError::InvalidConfig(
                "execution_mode must be IslandModel for IslandModel::new".into(),
            ));
        }

        ga_config.validate()?;
        let island_config = ga_config.island_config();
        island_config.validate()?;

        let mut islands = Vec::with_capacity(island_config.num_islands);
        for i in 0..island_config.num_islands {
            let mut config = ga_config.clone();
            config.random_seed = ga_config
                .random_seed
                .map(|seed| seed.wrapping_add(i as u64));
            islands.push(GeneticAlgorithm::new(config, fitness_fn.clone())?);
        }

        Ok(Self {
            island_config,
            islands,
        })
    }


    pub fn with_unified_configs(
        ga_configs: Vec<GaConfig>,
        fitness_fn: F,
    ) -> Result<Self, GaError> {
        let Some(first) = ga_configs.first() else {
            return Err(GaError::InvalidConfig(
                "ga_configs must contain at least one config".into(),
            ));
        };

        if first.execution_mode != EngineExecutionMode::IslandModel {
            return Err(GaError::InvalidConfig(
                "all configs must use EngineExecutionMode::IslandModel".into(),
            ));
        }

        let island_config = first.island_config();
        island_config.validate()?;

        if ga_configs.len() != island_config.num_islands {
            return Err(GaError::InvalidConfig(format!(
                "expected {} GA configs, got {}",
                island_config.num_islands,
                ga_configs.len()
            )));
        }

        for (index, config) in ga_configs.iter().enumerate() {
            config.validate()?;
            Self::validate_uniform_island_settings(&island_config, config, index)?;
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

    fn migrate(&mut self) {
        let k = self.island_config.migration_count;
        let n = self.islands.len();

        let emigrants: Vec<Vec<Individual>> = self
            .islands
            .iter()
            .map(|island| island.population.elite(k))
            .collect();

        for (src, src_emigrants) in emigrants.iter().enumerate() {
            let neighbors = self.island_config.topology.neighbors(src, n);
            for &dst in &neighbors {
                let island = &mut self.islands[dst];
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

    pub fn run(&mut self) -> Result<Vec<&RunStats>, GaError> {
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

            if global_generation < max_generations {
                self.migrate();
            }
        }

        Ok(self.islands.iter().map(|island| &island.stats).collect())
    }

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

    pub fn best_fitness(&self) -> Option<f64> {
        self.islands
            .iter()
            .filter_map(|island| island.stats.best_fitness)
            .max_by(|a, b| a.partial_cmp(b).expect("fitness comparison failed"))
    }
}

/// 统一引擎运行结果。
pub enum EngineRunResult<'a> {
    Single(&'a RunStats),
    Island(Vec<&'a RunStats>),
}

enum EngineBackend<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync + Clone,
{
    Single(GeneticAlgorithm<F>),
    Island(IslandModel<F>),
}

/// 单一入口引擎门面。
pub struct EvolutionEngine<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync + Clone,
{
    backend: EngineBackend<F>,
}

impl<F> EvolutionEngine<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync + Clone,
{
    pub fn new(config: GaConfig, fitness_fn: F) -> Result<Self, GaError> {
        let backend = match config.execution_mode {
            EngineExecutionMode::SinglePopulation => {
                EngineBackend::Single(GeneticAlgorithm::new(config, fitness_fn)?)
            }
            EngineExecutionMode::IslandModel => {
                EngineBackend::Island(IslandModel::new(config, fitness_fn)?)
            }
        };

        Ok(Self { backend })
    }

    pub fn run(&mut self) -> Result<EngineRunResult<'_>, GaError> {
        match &mut self.backend {
            EngineBackend::Single(engine) => Ok(EngineRunResult::Single(engine.run()?)),
            EngineBackend::Island(engine) => Ok(EngineRunResult::Island(engine.run()?)),
        }
    }

    pub fn best_solution(&self) -> Result<&Individual, GaError> {
        match &self.backend {
            EngineBackend::Single(engine) => engine.best_solution(),
            EngineBackend::Island(engine) => engine.best_solution(),
        }
    }

    pub fn best_fitness(&self) -> Option<f64> {
        match &self.backend {
            EngineBackend::Single(engine) => engine.stats.best_fitness,
            EngineBackend::Island(engine) => engine.best_fitness(),
        }
    }
}
