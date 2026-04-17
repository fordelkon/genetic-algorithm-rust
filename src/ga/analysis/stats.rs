use std::path::Path;

use crate::ga::analysis::{report::ExperimentSummary, visualize};
use crate::ga::core::{
    gene::GeneValue,
    pareto::ParetoSolution,
    population::Population,
};
use crate::ga::engine::config::OptimizationMode;
use crate::ga::error::GaError;
/// Runtime statistics tracking for GA evolution.

/// Per-generation metrics for NSGA-II runs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MultiObjectiveStats {
    pub front_0_size_per_generation: Vec<usize>,
    pub front_count_per_generation: Vec<usize>,
    pub best_front_per_generation: Vec<Vec<Vec<f64>>>,
    pub best_front_genes_per_generation: Vec<Vec<Vec<GeneValue>>>,
    pub final_pareto_front: Vec<ParetoSolution>,
}

/// Per-generation metrics and global-best snapshots.
#[derive(Debug, Clone, PartialEq)]
pub struct RunStats {
    /// Best fitness value for each generation.
    pub best_fitness_per_generation: Vec<f64>,
    /// Average fitness value for each generation.
    pub avg_fitness_per_generation: Vec<f64>,
    /// Fitness standard deviation for each generation.
    pub std_fitness_per_generation: Vec<f64>,
    /// Best genes for each generation.
    pub best_genes_per_generation: Vec<Vec<GeneValue>>,
    /// Global best genes observed so far.
    pub best_solution: Option<Vec<GeneValue>>,
    /// Global best fitness observed so far.
    pub best_fitness: Option<f64>,
    /// Optional NSGA-II metrics.
    pub multi_objective: Option<MultiObjectiveStats>,
}

impl Default for RunStats {
    fn default() -> Self {
        Self {
            best_fitness_per_generation: Vec::new(),
            avg_fitness_per_generation: Vec::new(),
            std_fitness_per_generation: Vec::new(),
            best_genes_per_generation: Vec::new(),
            best_solution: None,
            best_fitness: None,
            multi_objective: None,
        }
    }
}

impl RunStats {
    /// Records one generation of metrics from the current population.
    pub fn record(
        &mut self,
        population: &Population,
        optimization_mode: &OptimizationMode,
    ) -> Result<(), GaError> {
        match optimization_mode {
            OptimizationMode::SingleObjective => self.record_single(population),
            OptimizationMode::Nsga2 { .. } => self.record_multi(population),
        }
    }

    fn record_single(&mut self, population: &Population) -> Result<(), GaError> {
        let best = population.best()?;
        let avg = population.average_fitness()?;
        let std_dev = population.fitness_std_dev()?;

        self.best_fitness_per_generation.push(best.fitness_or_panic());
        self.avg_fitness_per_generation.push(avg);
        self.std_fitness_per_generation.push(std_dev);
        self.best_genes_per_generation.push(best.genes.clone());

        if self
            .best_fitness
            .is_none_or(|fitness| best.fitness_or_panic() > fitness)
        {
            self.best_fitness = Some(best.fitness_or_panic());
            self.best_solution = Some(best.genes.clone());
        }

        Ok(())
    }

    fn record_multi(&mut self, population: &Population) -> Result<(), GaError> {
        let front = population.pareto_front();
        let multi = self.multi_objective.get_or_insert_with(MultiObjectiveStats::default);
        multi.front_0_size_per_generation.push(front.len());

        let max_rank = population
            .individuals
            .iter()
            .filter_map(|individual| individual.rank)
            .max()
            .map(|rank| rank + 1)
            .unwrap_or(0);
        multi.front_count_per_generation.push(max_rank);
        multi.best_front_per_generation.push(
            front
                .iter()
                .map(|individual| individual.objectives_or_panic().to_vec())
                .collect(),
        );
        multi.best_front_genes_per_generation.push(
            front
                .iter()
                .map(|individual| individual.genes.clone())
                .collect(),
        );
        multi.final_pareto_front = front
            .iter()
            .map(ParetoSolution::from_individual)
            .collect::<Vec<_>>();
        Ok(())
    }

    /// Returns the latest recorded best fitness for single-objective runs.
    pub fn last_best(&self) -> Option<f64> {
        self.best_fitness_per_generation.last().copied()
    }

    /// Returns a compact single-objective summary of the run.
    pub fn summary(&self) -> Result<ExperimentSummary, GaError> {
        ExperimentSummary::from_stats(self)
    }

    /// Renders charts and markdown summary into an output directory.
    pub fn render_report<P: AsRef<Path>>(&self, output_dir: P) -> Result<(), GaError> {
        visualize::render_report(
            self,
            output_dir.as_ref(),
            &visualize::VisualizationOptions::default(),
        )
    }
}
