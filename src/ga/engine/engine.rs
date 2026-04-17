use std::{cmp::Ordering, sync::Arc};

use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::ga::analysis::stats::RunStats;
use crate::ga::core::{
    evaluation::{Evaluation, IntoEvaluation},
    gene::GeneValue,
    individual::Individual,
    pareto::ParetoSolution,
    population::Population,
};
use crate::ga::engine::config::{EngineConfig, OptimizationMode};
use crate::ga::error::GaError;
use crate::ga::{
    operators::crossover, operators::migration, operators::mutation, operators::selection,
    operators::stop,
};

pub use crate::ga::engine::config::MigrationType;
/// Engine backends and orchestration entry points.

type EvalFn = Arc<dyn Fn(&[GeneValue]) -> Evaluation + Send + Sync>;

/// Single-population execution kernel.
pub struct EngineKernel {
    pub config: EngineConfig,
    pub population: Population,
    pub generation: usize,
    pub stats: RunStats,
    evaluator: EvalFn,
    rng: StdRng,
}

impl EngineKernel {
    /// Creates a kernel from a validated configuration and evaluator.
    pub fn new<F, R>(config: EngineConfig, evaluator: F) -> Result<Self, GaError>
    where
        F: Fn(&[GeneValue]) -> R + Send + Sync + 'static,
        R: IntoEvaluation,
    {
        config.validate()?;
        let rng = Self::build_rng(config.random_seed);
        let evaluator = Arc::new(move |genes: &[GeneValue]| evaluator(genes).into_evaluation());

        Ok(Self {
            config,
            population: Population::empty(),
            generation: 0,
            stats: RunStats::default(),
            evaluator,
            rng,
        })
    }

    fn from_evaluator(config: EngineConfig, evaluator: EvalFn) -> Result<Self, GaError> {
        config.validate()?;
        let rng = Self::build_rng(config.random_seed);

        Ok(Self {
            config,
            population: Population::empty(),
            generation: 0,
            stats: RunStats::default(),
            evaluator,
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

    /// Evaluates all individuals and refreshes metadata for the configured mode.
    pub fn evaluate_population(&mut self) -> Result<(), GaError> {
        let evaluator = Arc::clone(&self.evaluator);
        let optimization_mode = self.config.optimization_mode.clone();
        self.population
            .evaluate_with(&optimization_mode, evaluator.as_ref())
    }

    /// Selects parents using the configured optimization mode.
    pub fn select_parents(&mut self) -> Result<Vec<Individual>, GaError> {
        match self.config.optimization_mode {
            OptimizationMode::SingleObjective => Ok(selection::select_parents(
                &self.population,
                &self.config.selection_type,
                self.config.num_parents_mating,
                &mut self.rng,
            )),
            OptimizationMode::Nsga2 { .. } => selection::select_nsga2_parents(
                &self.population,
                self.config.num_parents_mating,
                &mut self.rng,
            ),
        }
    }

    /// Produces offspring by applying crossover.
    pub fn crossover(&mut self, parents: &[Individual]) -> Vec<Individual> {
        let offspring_count = match self.config.optimization_mode {
            OptimizationMode::SingleObjective => self
                .config
                .population_size
                .saturating_sub(self.config.elitism_count),
            OptimizationMode::Nsga2 { .. } => self.config.population_size,
        };

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
            let evaluator = Arc::clone(&self.evaluator);
            offspring.par_iter_mut().for_each(|individual| {
                individual.set_evaluation(evaluator(&individual.genes));
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
        let parents = self.select_parents()?;
        let mut offspring = self.crossover(&parents);
        self.mutate(&mut offspring);

        match self.config.optimization_mode {
            OptimizationMode::SingleObjective => {
                let mut next_individuals = self.population.elite(self.config.elitism_count);
                next_individuals.extend(offspring);
                if next_individuals.len() > self.config.population_size {
                    next_individuals.truncate(self.config.population_size);
                }
                self.population = Population::new(next_individuals);
                self.evaluate_population()?;
            }
            OptimizationMode::Nsga2 { .. } => {
                let mut offspring_population = Population::new(offspring);
                offspring_population
                    .evaluate_with(&self.config.optimization_mode, self.evaluator.as_ref())?;

                let mut combined = self.population.individuals.clone();
                combined.extend(offspring_population.individuals);
                let mut combined_population = Population::new(combined);
                combined_population.assign_nsga2_metadata()?;

                let survivors = combined_population
                    .sorted_nsga2()?
                    .into_iter()
                    .take(self.config.population_size)
                    .collect::<Vec<_>>();
                self.population = Population::new(survivors);
                self.population.assign_nsga2_metadata()?;
            }
        }

        self.stats.record(&self.population, &self.config.optimization_mode)?;
        Ok(())
    }

    /// Runs evolution until the configured stop condition is met.
    pub fn run(&mut self) -> Result<&RunStats, GaError> {
        self.initialize_population()?;
        self.evaluate_population()?;
        self.stats.record(&self.population, &self.config.optimization_mode)?;

        loop {
            let should_stop = match self.config.optimization_mode {
                OptimizationMode::SingleObjective => stop::should_stop(
                    &self.config.stop_condition,
                    self.generation,
                    self.best_solution()?.fitness_or_panic(),
                    &self.stats,
                    self.config.num_generations,
                ),
                OptimizationMode::Nsga2 { .. } => {
                    self.generation >= self.config.num_generations
                }
            };

            if should_stop {
                break;
            }

            self.next_generation()?;
            self.generation += 1;
        }

        Ok(&self.stats)
    }

    /// Returns the current best individual for single-objective runs.
    pub fn best_solution(&self) -> Result<&Individual, GaError> {
        match self.config.optimization_mode {
            OptimizationMode::SingleObjective => self.population.best(),
            OptimizationMode::Nsga2 { .. } => Err(GaError::UnsupportedOperation(
                "best_solution is unavailable in NSGA-II mode; use pareto_front instead".into(),
            )),
        }
    }

    /// Returns the current Pareto front for NSGA-II runs.
    pub fn pareto_front(&self) -> Result<Vec<ParetoSolution>, GaError> {
        match self.config.optimization_mode {
            OptimizationMode::SingleObjective => Err(GaError::UnsupportedOperation(
                "pareto_front is unavailable in single-objective mode".into(),
            )),
            OptimizationMode::Nsga2 { .. } => Ok(self
                .population
                .pareto_front()
                .iter()
                .map(ParetoSolution::from_individual)
                .collect()),
        }
    }
}

/// Island model genetic algorithm.
pub struct IslandEngine {
    pub num_islands: usize,
    pub migration_count: usize,
    pub migration_interval: usize,
    pub migration_type: MigrationType,
    pub stats: RunStats,
    pub islands: Vec<EngineKernel>,
    optimization_mode: OptimizationMode,
}

impl IslandEngine {
    /// Builds an island model from one shared island configuration.
    pub fn new<F, R>(config: EngineConfig, evaluator: F) -> Result<Self, GaError>
    where
        F: Fn(&[GeneValue]) -> R + Send + Sync + 'static,
        R: IntoEvaluation,
    {
        let evaluator: EvalFn = Arc::new(move |genes: &[GeneValue]| evaluator(genes).into_evaluation());
        Self::from_evaluator(config, evaluator)
    }

    fn from_evaluator(config: EngineConfig, evaluator: EvalFn) -> Result<Self, GaError> {
        if config.num_islands <= 1 {
            return Err(GaError::InvalidConfig(
                "num_islands must be greater than 1 for IslandEngine::new".into(),
            ));
        }

        config.validate()?;

        let mut islands = Vec::with_capacity(config.num_islands);
        for i in 0..config.num_islands {
            let mut config_mut = config.clone();
            config_mut.random_seed = config.random_seed.map(|seed| seed.wrapping_add(i as u64));
            islands.push(EngineKernel::from_evaluator(
                config_mut,
                Arc::clone(&evaluator),
            )?);
        }

        Ok(Self {
            num_islands: config.num_islands,
            migration_count: config.migration_count,
            migration_interval: config.migration_interval,
            migration_type: config.migration_type.clone(),
            stats: RunStats::default(),
            islands,
            optimization_mode: config.optimization_mode,
        })
    }

    fn migrate(&mut self) -> Result<(), GaError> {
        migration::migrate(
            &mut self.islands,
            &self.migration_type,
            self.migration_count,
            &self.optimization_mode,
        );
        Ok(())
    }

    fn aggregate_stats(&self) -> Result<RunStats, GaError> {
        match &self.optimization_mode {
            OptimizationMode::SingleObjective => Ok(self
                .islands
                .iter()
                .map(|island| &island.stats)
                .max_by(|a, b| {
                    a.best_fitness
                        .unwrap_or(f64::NEG_INFINITY)
                        .partial_cmp(&b.best_fitness.unwrap_or(f64::NEG_INFINITY))
                        .unwrap_or(Ordering::Equal)
                })
                .cloned()
                .unwrap_or_default()),
            OptimizationMode::Nsga2 { .. } => {
                let mut combined = Population::new(
                    self.islands
                        .iter()
                        .flat_map(|island| island.population.individuals.clone())
                        .collect(),
                );
                combined.assign_nsga2_metadata()?;

                let mut aggregate = self
                    .islands
                    .first()
                    .map(|island| island.stats.clone())
                    .unwrap_or_default();
                let multi = aggregate.multi_objective.get_or_insert_default();
                multi.final_pareto_front = combined
                    .pareto_front()
                    .iter()
                    .map(ParetoSolution::from_individual)
                    .collect();
                Ok(aggregate)
            }
        }
    }

    /// Runs all islands and performs periodic migration.
    pub fn run(&mut self) -> Result<Vec<&RunStats>, GaError> {
        self.islands.par_iter_mut().try_for_each(|island| {
            island.initialize_population()?;
            island.evaluate_population()?;
            island
                .stats
                .record(&island.population, &island.config.optimization_mode)?;
            Ok::<(), GaError>(())
        })?;

        let max_generations = self
            .islands
            .iter()
            .map(|island| island.config.num_generations)
            .max()
            .unwrap_or(0);

        let mut global_generation = 0usize;
        while global_generation < max_generations {
            let steps = self
                .migration_interval
                .min(max_generations.saturating_sub(global_generation));

            for _ in 0..steps {
                self.islands.par_iter_mut().try_for_each(|island| {
                    let should_stop = match island.config.optimization_mode {
                        OptimizationMode::SingleObjective => stop::should_stop(
                            &island.config.stop_condition,
                            island.generation,
                            island.best_solution()?.fitness_or_panic(),
                            &island.stats,
                            island.config.num_generations,
                        ),
                        OptimizationMode::Nsga2 { .. } => {
                            island.generation >= island.config.num_generations
                        }
                    };

                    if !should_stop {
                        island.next_generation()?;
                        island.generation += 1;
                    }
                    Ok::<(), GaError>(())
                })?;
                global_generation += 1;
            }

            if global_generation < max_generations {
                self.migrate()?;
            }
        }

        self.stats = self.aggregate_stats()?;
        Ok(self.islands.iter().map(|island| &island.stats).collect())
    }

    /// Returns the global best solution and its island index across all islands.
    pub fn best_solution(&self) -> Result<(usize, &Individual), GaError> {
        match self.optimization_mode {
            OptimizationMode::SingleObjective => self
                .islands
                .iter()
                .enumerate()
                .filter_map(|(index, island)| island.population.best().ok().map(|best| (index, best)))
                .max_by(|(_, a), (_, b)| {
                    a.fitness_or_panic()
                        .partial_cmp(&b.fitness_or_panic())
                        .expect("fitness comparison failed")
                })
                .ok_or(GaError::EmptyPopulation),
            OptimizationMode::Nsga2 { .. } => Err(GaError::UnsupportedOperation(
                "best_solution is unavailable in NSGA-II mode; use pareto_front instead".into(),
            )),
        }
    }

    /// Returns the global best fitness and its island index across all islands.
    pub fn best_fitness(&self) -> Result<(usize, f64), GaError> {
        match self.optimization_mode {
            OptimizationMode::SingleObjective => Ok(self
                .islands
                .iter()
                .enumerate()
                .filter_map(|(index, island)| island.stats.best_fitness.map(|fitness| (index, fitness)))
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("fitness comparison failed")))
                ?.ok_or(GaError::EmptyPopulation),
            OptimizationMode::Nsga2 { .. } => Err(GaError::UnsupportedOperation(
                "best_fitness is unavailable in NSGA-II mode; use pareto_front instead".into(),
            )),
        }
    }

    /// Returns the global Pareto front across all islands.
    pub fn pareto_front(&self) -> Result<Vec<ParetoSolution>, GaError> {
        match self.optimization_mode {
            OptimizationMode::SingleObjective => Err(GaError::UnsupportedOperation(
                "pareto_front is unavailable in single-objective mode".into(),
            )),
            OptimizationMode::Nsga2 { .. } => {
                let mut combined = Population::new(
                    self.islands
                        .iter()
                        .flat_map(|island| island.population.individuals.clone())
                        .collect(),
                );
                combined.assign_nsga2_metadata()?;
                Ok(combined
                    .pareto_front()
                    .iter()
                    .map(ParetoSolution::from_individual)
                    .collect())
            }
        }
    }
}

/// Unified run result across engine backends.
pub enum EngineRunResult<'a> {
    Single(&'a RunStats),
    Island(Vec<&'a RunStats>),
}

enum EngineBackend {
    Single(EngineKernel),
    Island(IslandEngine),
}

/// Single entry-point facade that dispatches to the appropriate backend.
pub struct EvolutionEngine {
    pub stats: RunStats,
    backend: EngineBackend,
}

impl EvolutionEngine {
    /// Creates a new engine and infers backend from island count.
    pub fn new<F, R>(config: EngineConfig, evaluator: F) -> Result<Self, GaError>
    where
        F: Fn(&[GeneValue]) -> R + Send + Sync + 'static,
        R: IntoEvaluation,
    {
        let evaluator: EvalFn = Arc::new(move |genes: &[GeneValue]| evaluator(genes).into_evaluation());
        let backend = if config.num_islands > 1 {
            EngineBackend::Island(IslandEngine::from_evaluator(config, evaluator)?)
        } else {
            EngineBackend::Single(EngineKernel::from_evaluator(config, evaluator)?)
        };

        Ok(Self {
            stats: RunStats::default(),
            backend,
        })
    }

    /// Runs the underlying backend.
    pub fn run(&mut self) -> Result<EngineRunResult<'_>, GaError> {
        match &mut self.backend {
            EngineBackend::Single(engine) => {
                let stats = engine.run()?;
                self.stats = stats.clone();
                Ok(EngineRunResult::Single(stats))
            }
            EngineBackend::Island(engine) => {
                engine.run()?;
                self.stats = engine.stats.clone();
                Ok(EngineRunResult::Island(
                    engine.islands.iter().map(|island| &island.stats).collect(),
                ))
            }
        }
    }

    /// Returns the current best solution.
    pub fn best_solution(&self) -> Result<(usize, &Individual), GaError> {
        match &self.backend {
            EngineBackend::Single(engine) => engine.best_solution().map(|best| (0, best)),
            EngineBackend::Island(engine) => engine.best_solution(),
        }
    }

    /// Returns the current island index and best fitness for single-objective runs.
    pub fn best_fitness(&self) -> Result<(usize, f64), GaError> {
        match &self.backend {
            EngineBackend::Single(engine) => engine
                .stats
                .best_fitness
                .map(|fitness| (0, fitness))
                .ok_or(GaError::EmptyPopulation),
            EngineBackend::Island(engine) => engine.best_fitness(),
        }
    }

    /// Returns the current Pareto front for NSGA-II runs.
    pub fn pareto_front(&self) -> Result<Vec<ParetoSolution>, GaError> {
        match &self.backend {
            EngineBackend::Single(engine) => engine.pareto_front(),
            EngineBackend::Island(engine) => engine.pareto_front(),
        }
    }
}
