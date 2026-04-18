use std::cmp::Ordering;

use rand::Rng;

use crate::ga::core::{individual::Individual, population::Population};
use crate::ga::engine::config::OptimizationMode;
use crate::ga::error::GaError;
/// Parent selection operators.

/// Supported parent selection strategies.
#[derive(Debug, Clone)]
pub enum SelectionType {
    /// Selects the top-ranked individuals directly.
    SteadyState,

    /// Tournament selection with tournament size k.
    Tournament { k: usize },

    /// Roulette-wheel (fitness-proportionate) selection.
    RouletteWheel,

    /// Rank-based selection.
    Rank,

    /// Stochastic universal sampling.
    StochasticUniversalSampling,
}

/// Selects parent individuals according to the configured strategy.
pub fn select_parents(
    population: &Population,
    selection_type: &SelectionType,
    num_parents: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    match selection_type {
        SelectionType::SteadyState => steady_state_selection(population, num_parents),
        SelectionType::Tournament { k } => tournament_selection(population, num_parents, *k, rng),
        SelectionType::RouletteWheel => roulette_wheel_selection(population, num_parents, rng),
        SelectionType::Rank => rank_selection(population, num_parents, rng),
        SelectionType::StochasticUniversalSampling => {
            stochastic_universal_selection(population, num_parents, rng)
        }
    }
}

/// Selects parents using NSGA-II binary tournament rules.
pub fn select_nsga2_parents(
    population: &Population,
    num_parents: usize,
    rng: &mut impl Rng,
) -> Result<Vec<Individual>, GaError> {
    let len = population.len();
    if len == 0 {
        return Err(GaError::EmptyPopulation);
    }

    let mut parents = Vec::with_capacity(num_parents);
    for _ in 0..num_parents {
        let left = &population.individuals[rng.gen_range(0..len)];
        let right = &population.individuals[rng.gen_range(0..len)];
        parents.push(nsga2_better(left, right)?.clone());
    }

    Ok(parents)
}

/// Returns the population sorted by survivor priority for the active optimization mode.
pub fn sort_survivors(
    population: &Population,
    optimization_mode: &OptimizationMode,
) -> Result<Vec<Individual>, GaError> {
    match optimization_mode {
        OptimizationMode::SingleObjective => {
            let mut ranked = population.individuals.clone();
            ranked.sort_by(single_objective_ordering);
            Ok(ranked)
        }
        OptimizationMode::Nsga2 { .. } => Ok(sorted_population(population)),
    }
}

/// Returns the top-k survivors for the active optimization mode.
pub fn select_survivors(
    population: &Population,
    optimization_mode: &OptimizationMode,
    count: usize,
) -> Result<Vec<Individual>, GaError> {
    Ok(sort_survivors(population, optimization_mode)?
        .into_iter()
        .take(count)
        .collect())
}

/// Internal helper for steady-state selection.
fn steady_state_selection(population: &Population, num_parents: usize) -> Vec<Individual> {
    let mut ranked = population.individuals.clone();
    ranked.sort_by(single_objective_ordering);
    ranked.into_iter().take(num_parents).collect()
}

/// Internal helper for k-way tournament selection.
fn tournament_selection(
    population: &Population,
    num_parents: usize,
    k: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    let mut parents = Vec::with_capacity(num_parents);
    let len = population.len();

    for _ in 0..num_parents {
        let mut best: Option<&Individual> = None;

        for _ in 0..k {
            let candidate = &population.individuals[rng.gen_range(0..len)];
            if best
                .as_ref()
                .is_none_or(|current| candidate.fitness_or_panic() > current.fitness_or_panic())
            {
                best = Some(candidate);
            }
        }

        parents.push(best.expect("tournament should select a parent").clone());
    }

    parents
}

/// Internal helper for roulette-wheel parent selection.
fn roulette_wheel_selection(
    population: &Population,
    num_parents: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    let Some((weights, total)) = positive_selection_weights(population) else {
        return tournament_selection(population, num_parents, 2, rng);
    };

    let mut parents = Vec::with_capacity(num_parents);
    for _ in 0..num_parents {
        let mut threshold = rng.gen_range(0.0..total);
        let mut selected = &population.individuals[population.len() - 1];

        for (individual, weight) in population.individuals.iter().zip(weights.iter()) {
            threshold -= *weight;
            if threshold <= 0.0 {
                selected = individual;
                break;
            }
        }

        parents.push(selected.clone());
    }

    parents
}

/// Internal helper for stochastic universal sampling selection.
fn stochastic_universal_selection(
    population: &Population,
    num_parents: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    let Some((weights, total)) = positive_selection_weights(population) else {
        return tournament_selection(population, num_parents, 2, rng);
    };

    if num_parents == 0 {
        return Vec::new();
    }

    let step = total / num_parents as f64;
    let first_pointer = rng.gen_range(0.0..step);
    let mut parents = Vec::with_capacity(num_parents);
    let mut cumulative = 0.0;
    let mut selected_index = 0usize;

    for parent_idx in 0..num_parents {
        let pointer = first_pointer + step * parent_idx as f64;

        while selected_index < weights.len() - 1 && cumulative + weights[selected_index] < pointer {
            cumulative += weights[selected_index];
            selected_index += 1;
        }

        parents.push(population.individuals[selected_index].clone());
    }

    parents
}

/// Internal helper for rank-based parent selection.
fn rank_selection(
    population: &Population,
    num_parents: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    let mut ranked = population.individuals.clone();
    ranked.sort_by(single_objective_ordering);

    let weights = (1..=ranked.len())
        .rev()
        .map(|rank| rank as f64)
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    let mut parents = Vec::with_capacity(num_parents);

    for _ in 0..num_parents {
        let mut threshold = rng.gen_range(0.0..total);
        let mut selected = ranked
            .last()
            .expect("ranked population should not be empty");

        for (individual, weight) in ranked.iter().zip(weights.iter()) {
            threshold -= *weight;
            if threshold <= 0.0 {
                selected = individual;
                break;
            }
        }

        parents.push(selected.clone());
    }

    parents
}

/// Builds strictly positive fitness-based weights for probability selection operators.
fn positive_selection_weights(population: &Population) -> Option<(Vec<f64>, f64)> {
    let min_fitness = population
        .individuals
        .iter()
        .map(Individual::fitness_or_panic)
        .fold(f64::INFINITY, f64::min);

    let shift = if min_fitness <= 0.0 {
        -min_fitness + 1e-9
    } else {
        0.0
    };

    let weights = population
        .individuals
        .iter()
        .map(|individual| individual.fitness_or_panic() + shift)
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();

    if total <= 0.0 {
        None
    } else {
        Some((weights, total))
    }
}

/// Compares two individuals using NSGA-II priority: lower rank first, then higher crowding distance.
fn nsga2_ordering(left: &Individual, right: &Individual) -> Ordering {
    let rank_cmp = left
        .rank
        .unwrap_or(usize::MAX)
        .cmp(&right.rank.unwrap_or(usize::MAX));
    if rank_cmp != Ordering::Equal {
        return rank_cmp;
    }

    right
        .crowding_distance
        .unwrap_or(f64::NEG_INFINITY)
        .partial_cmp(&left.crowding_distance.unwrap_or(f64::NEG_INFINITY))
        .unwrap_or(Ordering::Equal)
}

/// Returns a cloned list ordered by NSGA-II priority.
fn sorted_population(population: &Population) -> Vec<Individual> {
    let mut sorted = population.individuals.clone();
    sorted.sort_by(nsga2_ordering);
    sorted
}

fn single_objective_ordering(left: &Individual, right: &Individual) -> Ordering {
    right
        .fitness_or_panic()
        .partial_cmp(&left.fitness_or_panic())
        .expect("fitness comparison failed")
}

/// Chooses the better individual under NSGA-II tournament rules.
fn nsga2_better<'a>(
    left: &'a Individual,
    right: &'a Individual,
) -> Result<&'a Individual, GaError> {
    let left_rank = left
        .rank
        .ok_or_else(|| GaError::UnsupportedOperation("NSGA-II rank metadata is missing".into()))?;
    let right_rank = right
        .rank
        .ok_or_else(|| GaError::UnsupportedOperation("NSGA-II rank metadata is missing".into()))?;

    if left_rank != right_rank {
        return Ok(if left_rank < right_rank { left } else { right });
    }

    let left_distance = left.crowding_distance.unwrap_or(f64::NEG_INFINITY);
    let right_distance = right.crowding_distance.unwrap_or(f64::NEG_INFINITY);

    if left_distance >= right_distance {
        Ok(left)
    } else {
        Ok(right)
    }
}
