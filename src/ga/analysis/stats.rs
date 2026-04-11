use std::path::Path;

use crate::ga::analysis::{report::ExperimentSummary, visualize};
use crate::ga::core::{gene::GeneValue, population::Population};
use crate::ga::error::GaError;
/// Runtime statistics tracking for GA evolution.

/// Per-generation metrics and global-best snapshots.
#[derive(Debug, Clone, Default)]
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
}

impl RunStats {
    /// Records one generation of metrics from the current population.
    pub fn record(&mut self, population: &Population) -> Result<(), GaError> {
        let best = population.best()?;
        let avg = population.average_fitness()?;
        let std_dev = population.fitness_std_dev()?;

        self.best_fitness_per_generation
            .push(best.fitness_or_panic());
        self.avg_fitness_per_generation.push(avg);
        self.std_fitness_per_generation.push(std_dev);
        self.best_genes_per_generation.push(best.genes.clone());

        if self
            .best_fitness
            .is_none_or(|fitness| best.fitness_or_panic() > fitness)
        {
            self.best_fitness = best.fitness;
            self.best_solution = Some(best.genes.clone());
        }

        Ok(())
    }

    /// Returns the latest recorded best fitness.
    pub fn last_best(&self) -> Option<f64> {
        self.best_fitness_per_generation.last().copied()
    }

    /// Returns a compact summary of the run.
    /// # Errors
    pub fn summary(&self) -> Result<ExperimentSummary, GaError> {
        ExperimentSummary::from_stats(self)
    }

    /// Renders charts and markdown summary into an output directory.
    ///
    /// - `fitness_history.svg`
    /// - `best_genes_final.svg`
    /// - `best_genes_trajectory.svg`
    /// - `summary.md`
    /// # Errors
    pub fn render_report<P: AsRef<Path>>(&self, output_dir: P) -> Result<(), GaError> {
        visualize::render_report(
            self,
            output_dir.as_ref(),
            &visualize::VisualizationOptions::default(),
        )
    }
}
