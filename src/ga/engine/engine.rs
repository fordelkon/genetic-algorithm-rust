use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::ga::analysis::stats::RunStats;
use crate::ga::core::{gene::GeneValue, individual::Individual, population::Population};
use crate::ga::engine::config::EngineConfig;
use crate::ga::error::GaError;
use crate::ga::{
    operators::crossover, operators::migrate, operators::mutation, operators::selection,
    operators::stop,
};

pub use crate::ga::engine::config::MigrationType;
/// Engine backends and orchestration entry points.

/// Single-population execution kernel.
pub struct EngineKernel<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync,
{
    pub config: EngineConfig,
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
    /// Creates a kernel from a validated configuration and fitness function.
    pub fn new(config: EngineConfig, fitness_fn: F) -> Result<Self, GaError> {
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

    /// Builds a deterministic RNG when seed is provided.
    pub fn build_rng(seed: Option<u64>) -> StdRng {
        match seed {
            Some(value) => StdRng::seed_from_u64(value),
            None => StdRng::seed_from_u64(rand::random()),
        }
    }

    /// Initializes the first population by sampling genes.
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

    /// Evaluates fitness for all individuals in parallel.
    pub fn evaluate_population(&mut self) {
        self.population
            .individuals
            .par_iter_mut()
            .for_each(|individual| {
                let fitness = (self.fitness_fn)(&individual.genes);
                individual.fitness = Some(fitness);
            });
    }

    /// Selects parents using the configured selection policy.
    pub fn select_parents(&mut self) -> Vec<Individual> {
        selection::select_parents(
            &self.population,
            &self.config.selection_type,
            self.config.num_parents_mating,
            &mut self.rng,
        )
    }

    /// Produces offspring by applying crossover.
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

    /// Applies mutation to offspring.
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

    /// Advances one generation and records run statistics.
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

    /// Runs evolution until the configured stop condition is met.
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

    /// Returns the current best individual.
    pub fn best_solution(&self) -> Result<&Individual, GaError> {
        self.population.best()
    }
}

/// Island model genetic algorithm.
pub struct IslandEngine<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync + Clone,
{
    pub num_islands: usize,
    pub migration_count: usize,
    pub migration_interval: usize,
    pub migration_type: MigrationType,
    pub islands: Vec<EngineKernel<F>>,
}

impl<F> IslandEngine<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync + Clone,
{
    /// Builds an island model from one shared island configuration.
    pub fn new(config: EngineConfig, fitness_fn: F) -> Result<Self, GaError> {
        if config.num_islands <= 1 {
            return Err(GaError::InvalidConfig(
                "num_islands must be greater than 1 for IslandEngine::new".into(),
            ));
        }

        config.validate()?;

        let mut islands = Vec::with_capacity(config.num_islands);
        for i in 0..config.num_islands {
            let mut config_mut = config.clone();
            config_mut.random_seed = config
                .random_seed
                .map(|seed| seed.wrapping_add(i as u64));
            islands.push(EngineKernel::new(config_mut, fitness_fn.clone())?);
        }

        Ok(Self {
            num_islands: config.num_islands,
            migration_count: config.migration_count,
            migration_interval: config.migration_interval,
            migration_type: config.migration_type.clone(),
            islands,
        })
    }

    fn migrate(&mut self) {
        let k = self.migration_count;
        let n = self.islands.len();

        let emigrants: Vec<Vec<Individual>> = self
            .islands
            .iter()
            .map(|island| island.population.elite(k))
            .collect();

        for (src, src_emigrants) in emigrants.iter().enumerate() {
            let neighbors = migrate::migrate(&self.migration_type, src, n);
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

    /// Runs all islands and performs periodic migration.
    pub fn run(&mut self) -> Result<Vec<&RunStats>, GaError>
    where
        F: Send,
    {
        self.islands.par_iter_mut().try_for_each(|island| {
            island.initialize_population()?;
            island.evaluate_population();
            island.stats.record(&island.population)?;
            Ok::<(), GaError>(())
        })?;

        let max_generations = self
            .islands
            .iter()
            .map(|island| island.config.num_generations)
            .max()
            .unwrap_or(0);

        let mut global_generation = 0;

        while global_generation < max_generations {
            let steps = self
                .migration_interval
                .min(max_generations - global_generation);

            for _ in 0..steps {
                self.islands.par_iter_mut().try_for_each(|island| {
                    if island.generation < island.config.num_generations {
                        island.next_generation()?;
                        island.generation += 1;
                    }
                    Ok::<(), GaError>(())
                })?;
                global_generation += 1;
            }

            if global_generation < max_generations {
                self.migrate();
            }
        }

        Ok(self.islands.iter().map(|island| &island.stats).collect())
    }

    /// Returns the global best solution and its island index across all islands.
    pub fn best_solution(&self) -> Result<(usize, &Individual), GaError> {
        self.islands
            .iter()
            .enumerate()
            .filter_map(|(index, island)| island.best_solution().ok().map(|best| (index, best)))
            .max_by(|(_, a), (_, b)| {
                a.fitness_or_panic()
                    .partial_cmp(&b.fitness_or_panic())
                    .expect("fitness comparison failed")
            })
            .ok_or(GaError::EmptyPopulation)
    }

    /// Returns the global best fitness and its island index across all islands.
    pub fn best_fitness(&self) -> Option<(usize, f64)> {
        self.islands
            .iter()
            .enumerate()
            .filter_map(|(index, island)| island.stats.best_fitness.map(|fitness| (index, fitness)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("fitness comparison failed"))
    }
}

/// Unified run result across engine backends.
pub enum EngineRunResult<'a> {
    Single(&'a RunStats),
    Island(Vec<&'a RunStats>),
}

enum EngineBackend<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync + Clone,
{
    Single(EngineKernel<F>),
    Island(IslandEngine<F>),
}

/// Single entry-point facade that dispatches to the appropriate backend.
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
    /// Creates a new engine and infers backend from island count.
    pub fn new(config: EngineConfig, fitness_fn: F) -> Result<Self, GaError> {
        let backend = if config.num_islands > 1 {
            EngineBackend::Island(IslandEngine::new(config, fitness_fn)?)
        } else {
            EngineBackend::Single(EngineKernel::new(config, fitness_fn)?)
        };

        Ok(Self { backend })
    }

    /// Runs the underlying backend.
    pub fn run(&mut self) -> Result<EngineRunResult<'_>, GaError>
    where
        F: Send,
    {
        match &mut self.backend {
            EngineBackend::Single(engine) => Ok(EngineRunResult::Single(engine.run()?)),
            EngineBackend::Island(engine) => Ok(EngineRunResult::Island(engine.run()?)),
        }
    }

    /// Returns the current best solution.
    pub fn best_solution(&self) -> Result<&Individual, GaError> {
        match &self.backend {
            EngineBackend::Single(engine) => engine.best_solution(),
            EngineBackend::Island(engine) => engine.best_solution().map(|(_, best)| best),
        }
    }

    /// Returns the current best fitness.
    pub fn best_fitness(&self) -> Option<f64> {
        match &self.backend {
            EngineBackend::Single(engine) => engine.stats.best_fitness,
            EngineBackend::Island(engine) => engine.best_fitness().map(|(_, fitness)| fitness),
        }
    }
}
