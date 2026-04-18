use std::cmp::Ordering;
use std::collections::HashMap;

use rayon::prelude::*;

use crate::ga::core::{
    evaluation::Evaluation, gene::GeneValue, individual::Individual, population::Population,
};
use crate::ga::engine::config::OptimizationMode;
use crate::ga::error::GaError;

/// Dominance relationship between two individuals in multi-objective minimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dominance {
    /// Left individual dominates the right individual.
    LeftDominates,

    /// Right individual dominates the left individual.
    RightDominates,

    /// Neither individual dominates the other.
    Neither,
}

/// Determines Pareto dominance between two evaluated individuals for minimization objectives.
fn dominates_minimizing(left: &Individual, right: &Individual) -> Result<Dominance, GaError> {
    let left_values = left
        .evaluation
        .as_ref()
        .ok_or(GaError::UnevaluatedFitness)?
        .as_multi()?;
    let right_values = right
        .evaluation
        .as_ref()
        .ok_or(GaError::UnevaluatedFitness)?
        .as_multi()?;

    if left_values.len() != right_values.len() {
        return Err(GaError::InvalidConfig(
            "all NSGA-II evaluations must have the same objective count".into(),
        ));
    }

    let left_not_worse = left_values
        .iter()
        .zip(right_values.iter())
        .all(|(l, r)| l <= r);
    let right_not_worse = left_values
        .iter()
        .zip(right_values.iter())
        .all(|(l, r)| l >= r);
    let left_strictly_better = left_values
        .iter()
        .zip(right_values.iter())
        .any(|(l, r)| l < r);
    let right_strictly_better = left_values
        .iter()
        .zip(right_values.iter())
        .any(|(l, r)| l > r);

    Ok(if left_not_worse && left_strictly_better {
        Dominance::LeftDominates
    } else if right_not_worse && right_strictly_better {
        Dominance::RightDominates
    } else {
        Dominance::Neither
    })
}

/// Evaluates all individuals and refreshes NSGA-II metadata for the population.
pub fn evaluate_population_with<F>(
    population: &mut Population,
    optimization_mode: &OptimizationMode,
    evaluator: &F,
) -> Result<(), GaError>
where
    F: Fn(&[GeneValue]) -> Evaluation + Send + Sync + ?Sized,
{
    population
        .individuals
        .par_iter_mut()
        .try_for_each(|individual| -> Result<(), GaError> {
            let evaluation = evaluator(&individual.genes);
            evaluation.validate_for_mode(optimization_mode)?;
            individual.set_evaluation(evaluation);
            Ok(())
        })?;

    if matches!(optimization_mode, OptimizationMode::Nsga2 { .. }) {
        assign_population_metadata(population)?;
    }

    Ok(())
}

/// Recomputes NSGA-II ranks and crowding distances in-place.
pub fn assign_population_metadata(population: &mut Population) -> Result<Vec<Vec<usize>>, GaError> {
    if population.is_empty() {
        return Ok(Vec::new());
    }

    let mut domination_counts = vec![0usize; population.len()];
    let mut dominates = vec![Vec::new(); population.len()];
    let mut fronts: Vec<Vec<usize>> = Vec::new();

    for i in 0..population.len() {
        population.individuals[i].rank = None;
        population.individuals[i].crowding_distance = Some(0.0);

        for j in 0..population.len() {
            if i == j {
                continue;
            }

            match dominates_minimizing(&population.individuals[i], &population.individuals[j])? {
                Dominance::LeftDominates => dominates[i].push(j),
                Dominance::RightDominates => domination_counts[i] += 1,
                Dominance::Neither => {}
            }
        }

        if domination_counts[i] == 0 {
            population.individuals[i].rank = Some(0);
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
                    population.individuals[dominated].rank = Some(front_index + 1);
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
        assign_crowding_distance(population, front)?;
    }

    Ok(fronts)
}

fn assign_crowding_distance(population: &mut Population, front: &[usize]) -> Result<(), GaError> {
    if front.is_empty() {
        return Ok(());
    }

    if front.len() <= 2 {
        for &index in front {
            population.individuals[index].crowding_distance = Some(f64::INFINITY);
        }
        return Ok(());
    }

    for &index in front {
        population.individuals[index].crowding_distance = Some(0.0);
    }

    let objective_count = match population.individuals[front[0]]
        .evaluation
        .as_ref()
        .ok_or(GaError::UnevaluatedFitness)?
    {
        Evaluation::Multi(values) => values.len(),
        Evaluation::Single(_) => {
            return Err(GaError::UnsupportedOperation(
                "NSGA-II crowding distance requires multi-objective evaluations".into(),
            ));
        }
    };

    let mut global_ranges = Vec::with_capacity(objective_count);
    for objective_index in 0..objective_count {
        let mut min_value = f64::INFINITY;
        let mut max_value = f64::NEG_INFINITY;

        for individual in &population.individuals {
            let value = individual.objectives_or_panic()[objective_index];
            min_value = min_value.min(value);
            max_value = max_value.max(value);
        }

        global_ranges.push((min_value, max_value));
    }

    for objective_index in 0..objective_count {
        let mut ranked = front.to_vec();
        ranked.sort_by(|left, right| {
            population.individuals[*left].objectives_or_panic()[objective_index]
                .partial_cmp(&population.individuals[*right].objectives_or_panic()[objective_index])
                .unwrap_or(Ordering::Equal)
        });

        let first = ranked[0];
        let last = ranked[ranked.len() - 1];
        population.individuals[first].crowding_distance = Some(f64::INFINITY);
        population.individuals[last].crowding_distance = Some(f64::INFINITY);

        let (min_value, max_value) = global_ranges[objective_index];
        let range = max_value - min_value;

        if range.abs() <= f64::EPSILON {
            continue;
        }

        for window in ranked.windows(3) {
            let prev = population.individuals[window[0]].objectives_or_panic()[objective_index];
            let next = population.individuals[window[2]].objectives_or_panic()[objective_index];
            let middle = window[1];

            let distance = population.individuals[middle]
                .crowding_distance
                .unwrap_or(0.0);
            if !distance.is_infinite() {
                population.individuals[middle].crowding_distance =
                    Some(distance + (next - prev) / range);
            }
        }
    }

    penalize_duplicate_objectives(population, front);

    Ok(())
}

fn penalize_duplicate_objectives(population: &mut Population, front: &[usize]) {
    let mut duplicate_groups: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();

    for &index in front {
        let key = population.individuals[index]
            .objectives_or_panic()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        duplicate_groups.entry(key).or_default().push(index);
    }

    for duplicates in duplicate_groups.into_values() {
        if duplicates.len() <= 1 {
            continue;
        }

        let representative = duplicates
            .iter()
            .copied()
            .max_by(|left, right| {
                population.individuals[*left]
                    .crowding_distance
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(
                        &population.individuals[*right]
                            .crowding_distance
                            .unwrap_or(f64::NEG_INFINITY),
                    )
                    .then_with(|| right.cmp(left))
            })
            .expect("duplicate group should not be empty");

        for index in duplicates {
            if index != representative {
                population.individuals[index].crowding_distance = Some(0.0);
            }
        }
    }
}
