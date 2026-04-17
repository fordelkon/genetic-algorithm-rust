use std::cmp::Ordering;

use rayon::prelude::*;

use crate::ga::core::{
    evaluation::Evaluation,
    gene::GeneValue,
    individual::Individual,
};
use crate::ga::engine::config::OptimizationMode;
use crate::ga::error::GaError;
use crate::ga::operators::nsga2;
/// Population container and fitness aggregation helpers.

/// Collection of individuals in one generation.
#[derive(Debug, Clone)]
pub struct Population {
    /// Individuals in the current population.
    pub individuals: Vec<Individual>,
}

impl Population {
    /// Creates a population from an individual list.
    pub fn new(individuals: Vec<Individual>) -> Self {
        Self { individuals }
    }

    /// Creates an empty population.
    pub fn empty() -> Self {
        Self {
            individuals: Vec::new(),
        }
    }

    /// Returns the number of individuals.
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Returns whether the population is empty.
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// Evaluates all individuals and refreshes NSGA-II metadata when needed.
    pub fn evaluate_with<F>(
        &mut self,
        optimization_mode: &OptimizationMode,
        evaluator: &F,
    ) -> Result<(), GaError>
    where
        F: Fn(&[GeneValue]) -> Evaluation + Send + Sync + ?Sized,
    {
        self.individuals
            .par_iter_mut()
            .try_for_each(|individual| -> Result<(), GaError> {
                let evaluation = evaluator(&individual.genes);
                evaluation.validate_for_mode(optimization_mode)?;
                individual.set_evaluation(evaluation);
                Ok(())
            })?;

        if matches!(optimization_mode, OptimizationMode::Nsga2 { .. }) {
            self.assign_nsga2_metadata()?;
        }

        Ok(())
    }

    /// Returns the highest-fitness individual in single-objective mode.
    pub fn best(&self) -> Result<&Individual, GaError> {
        self.individuals
            .iter()
            .max_by(|left, right| {
                left.fitness_or_panic()
                    .partial_cmp(&right.fitness_or_panic())
                    .expect("fitness comparison failed")
            })
            .ok_or(GaError::EmptyPopulation)
    }

    /// Returns clones of the top-k elite individuals in single-objective mode.
    pub fn elite(&self, count: usize) -> Vec<Individual> {
        let mut sorted = self.individuals.clone();
        sorted.sort_by(|left, right| {
            right
                .fitness_or_panic()
                .partial_cmp(&left.fitness_or_panic())
                .expect("fitness comparison failed")
        });
        sorted.into_iter().take(count).collect()
    }

    /// Sorts the population by fitness in descending order.
    pub fn sort_by_fitness_desc(&mut self) {
        self.individuals.sort_by(|left, right| {
            right
                .fitness_or_panic()
                .partial_cmp(&left.fitness_or_panic())
                .expect("fitness comparison failed")
        });
    }

    /// Returns average fitness across all individuals.
    pub fn average_fitness(&self) -> Result<f64, GaError> {
        if self.is_empty() {
            return Err(GaError::EmptyPopulation);
        }

        let total = self
            .individuals
            .iter()
            .map(Individual::fitness_or_panic)
            .sum::<f64>();
        Ok(total / self.len() as f64)
    }

    /// Returns population standard deviation of fitness.
    pub fn fitness_std_dev(&self) -> Result<f64, GaError> {
        let mean = self.average_fitness()?;
        let variance = self
            .individuals
            .iter()
            .map(|individual| {
                let delta = individual.fitness_or_panic() - mean;
                delta * delta
            })
            .sum::<f64>()
            / self.len() as f64;
        Ok(variance.sqrt())
    }

    /// Recomputes NSGA-II ranks and crowding distances in-place.
    pub fn assign_nsga2_metadata(&mut self) -> Result<Vec<Vec<usize>>, GaError> {
        if self.is_empty() {
            return Ok(Vec::new());
        }

        let mut domination_counts = vec![0usize; self.len()];
        let mut dominates = vec![Vec::new(); self.len()];
        let mut fronts: Vec<Vec<usize>> = Vec::new();

        for i in 0..self.len() {
            self.individuals[i].rank = None;
            self.individuals[i].crowding_distance = Some(0.0);

            for j in 0..self.len() {
                if i == j {
                    continue;
                }

                match nsga2::dominates_minimizing(&self.individuals[i], &self.individuals[j])? {
                    nsga2::Dominance::LeftDominates => dominates[i].push(j),
                    nsga2::Dominance::RightDominates => domination_counts[i] += 1,
                    nsga2::Dominance::Neither => {}
                }
            }

            if domination_counts[i] == 0 {
                self.individuals[i].rank = Some(0);
                if fronts.is_empty() {
                    fronts.push(Vec::new());
                }
                fronts[0].push(i);
            }
        }

        let mut front_index = 0usize;
        while front_index < fronts.len() {
            let mut next_front = Vec::new();

            for &current in &fronts[front_index] {
                for &dominated in &dominates[current] {
                    domination_counts[dominated] -= 1;
                    if domination_counts[dominated] == 0 {
                        self.individuals[dominated].rank = Some(front_index + 1);
                        next_front.push(dominated);
                    }
                }
            }

            if !next_front.is_empty() {
                fronts.push(next_front);
            }
            front_index += 1;
        }

        for front in &fronts {
            self.assign_crowding_distance(front)?;
        }

        Ok(fronts)
    }

    /// Returns a sorted list of NSGA-II survivors by rank then crowding distance.
    pub fn sorted_nsga2(&self) -> Result<Vec<Individual>, GaError> {
        let mut sorted = self.individuals.clone();
        sorted.sort_by(nsga2::nsga2_ordering);
        Ok(sorted)
    }

    /// Returns the first Pareto front as cloned individuals.
    pub fn pareto_front(&self) -> Vec<Individual> {
        self.individuals
            .iter()
            .filter(|individual| individual.rank == Some(0))
            .cloned()
            .collect()
    }

    /// Internal helper that computes NSGA-II crowding distance for one Pareto front.
    fn assign_crowding_distance(&mut self, front: &[usize]) -> Result<(), GaError> {
        if front.is_empty() {
            return Ok(());
        }

        if front.len() <= 2 {
            for &index in front {
                self.individuals[index].crowding_distance = Some(f64::INFINITY);
            }
            return Ok(());
        }

        for &index in front {
            self.individuals[index].crowding_distance = Some(0.0);
        }

        let objective_count = match self.individuals[front[0]]
            .evaluation
            .as_ref()
            .ok_or(GaError::UnevaluatedFitness)?
        {
            Evaluation::Multi(values) => values.len(),
            Evaluation::Single(_) => {
                return Err(GaError::UnsupportedOperation(
                    "NSGA-II crowding distance requires multi-objective evaluations".into(),
                ))
            }
        };

        for objective_index in 0..objective_count {
            let mut ranked = front.to_vec();
            ranked.sort_by(|left, right| {
                self.individuals[*left].objectives_or_panic()[objective_index]
                    .partial_cmp(&self.individuals[*right].objectives_or_panic()[objective_index])
                    .unwrap_or(Ordering::Equal)
            });

            let first = ranked[0];
            let last = ranked[ranked.len() - 1];
            self.individuals[first].crowding_distance = Some(f64::INFINITY);
            self.individuals[last].crowding_distance = Some(f64::INFINITY);

            let min_value = self.individuals[first].objectives_or_panic()[objective_index];
            let max_value = self.individuals[last].objectives_or_panic()[objective_index];
            let range = max_value - min_value;

            if range.abs() <= f64::EPSILON {
                continue;
            }

            for window in ranked.windows(3) {
                let prev = self.individuals[window[0]].objectives_or_panic()[objective_index];
                let next = self.individuals[window[2]].objectives_or_panic()[objective_index];
                let middle = window[1];

                let distance = self.individuals[middle].crowding_distance.unwrap_or(0.0);
                if !distance.is_infinite() {
                    self.individuals[middle].crowding_distance = Some(distance + (next - prev) / range);
                }
            }
        }

        Ok(())
    }
}
