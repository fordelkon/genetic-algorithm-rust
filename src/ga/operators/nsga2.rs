use std::cmp::Ordering;

use crate::ga::core::{
    individual::Individual,
};
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
pub fn dominates_minimizing(left: &Individual, right: &Individual) -> Result<Dominance, GaError> {
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

    let left_not_worse = left_values.iter().zip(right_values.iter()).all(|(l, r)| l <= r);
    let right_not_worse = left_values.iter().zip(right_values.iter()).all(|(l, r)| l >= r);
    let left_strictly_better = left_values.iter().zip(right_values.iter()).any(|(l, r)| l < r);
    let right_strictly_better = left_values.iter().zip(right_values.iter()).any(|(l, r)| l > r);

    Ok(if left_not_worse && left_strictly_better {
        Dominance::LeftDominates
    } else if right_not_worse && right_strictly_better {
        Dominance::RightDominates
    } else {
        Dominance::Neither
    })
}

/// Compares two individuals using NSGA-II priority: lower rank first, then higher crowding distance.
pub fn nsga2_ordering(left: &Individual, right: &Individual) -> Ordering {
    let rank_cmp = left.rank.unwrap_or(usize::MAX).cmp(&right.rank.unwrap_or(usize::MAX));
    if rank_cmp != Ordering::Equal {
        return rank_cmp;
    }

    right
        .crowding_distance
        .unwrap_or(f64::NEG_INFINITY)
        .partial_cmp(&left.crowding_distance.unwrap_or(f64::NEG_INFINITY))
        .unwrap_or(Ordering::Equal)
}
