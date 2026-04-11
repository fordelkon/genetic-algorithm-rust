use rand::Rng;

use crate::ga::core::{gene::GeneValue, individual::Individual};
/// Crossover operators for offspring generation.

/// Supported crossover strategies.
#[derive(Debug, Clone)]
pub enum CrossoverType {
    /// No crossover; offspring clones parent A.
    None,

    /// Single-point crossover.
    SinglePoint,

    /// Two-point crossover.
    TwoPoint,

    /// Uniform crossover.
    Uniform,
}

/// Produces offspring from selected parents using the configured crossover operator.
pub fn crossover(
    parents: &[Individual],
    crossover_type: &CrossoverType,
    crossover_probability: f64,
    offspring_count: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    let mut offspring = Vec::with_capacity(offspring_count);

    for idx in 0..offspring_count {
        let parent1 = &parents[idx % parents.len()];
        let parent2 = &parents[(idx + 1) % parents.len()];

        let genes = if rng.gen_bool(crossover_probability) {
            match crossover_type {
                CrossoverType::None => parent1.genes.clone(),
                CrossoverType::SinglePoint => single_point(&parent1.genes, &parent2.genes, rng),
                CrossoverType::TwoPoint => two_point(&parent1.genes, &parent2.genes, rng),
                CrossoverType::Uniform => uniform(&parent1.genes, &parent2.genes, rng),
            }
        } else {
            parent1.genes.clone()
        };

        offspring.push(Individual::new(genes));
    }

    offspring
}

fn single_point(
    parent1: &[GeneValue],
    parent2: &[GeneValue],
    rng: &mut impl Rng,
) -> Vec<GeneValue> {
    if parent1.len() <= 1 {
        return parent1.to_vec();
    }

    let split = rng.gen_range(1..parent1.len());
    parent1[..split]
        .iter()
        .chain(parent2[split..].iter())
        .cloned()
        .collect()
}

/// Internal helper for two-point crossover.
fn two_point(parent1: &[GeneValue], parent2: &[GeneValue], rng: &mut impl Rng) -> Vec<GeneValue> {
    if parent1.len() <= 2 {
        return single_point(parent1, parent2, rng);
    }

    let point1 = rng.gen_range(1..parent1.len() - 1);
    let point2 = rng.gen_range(point1 + 1..parent1.len());
    let mut child = Vec::with_capacity(parent1.len());

    child.extend_from_slice(&parent1[..point1]);
    child.extend_from_slice(&parent2[point1..point2]);
    child.extend_from_slice(&parent1[point2..]);
    child
}

/// Internal helper for uniform crossover.
fn uniform(parent1: &[GeneValue], parent2: &[GeneValue], rng: &mut impl Rng) -> Vec<GeneValue> {
    parent1
        .iter()
        .zip(parent2.iter())
        .map(|(left, right)| {
            if rng.gen_bool(0.5) {
                left.clone()
            } else {
                right.clone()
            }
        })
        .collect()
}
