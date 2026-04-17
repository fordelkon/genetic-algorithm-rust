use rand::{Rng, seq::SliceRandom};

use crate::ga::core::individual::Individual;
use crate::ga::engine::config::EngineConfig;
/// Mutation operators for exploration and diversity maintenance.

/// Supported mutation strategies.
#[derive(Debug, Clone)]
pub enum MutationType {
    /// No mutation.
    None,

    /// Replaces selected genes with freshly sampled values.
    RandomReset { min: f64, max: f64 },

    /// Adds a random delta to selected genes.
    RandomPerturbation { min_delta: f64, max_delta: f64 },

    /// Adaptive random reset based on relative fitness quality.
    AdaptiveRandomReset {
        min: f64,
        max: f64,
        low_quality_num_genes: usize,
        high_quality_num_genes: usize,
    },

    /// Adaptive random perturbation based on relative fitness quality.
    AdaptiveRandomPerturbation {
        min_delta: f64,
        max_delta: f64,
        low_quality_num_genes: usize,
        high_quality_num_genes: usize,
    },

    /// Swaps two gene positions.
    Swap,

    /// Shuffles a contiguous gene segment.
    Scramble,

    /// Reverses a contiguous gene segment.
    Inversion,
}

/// Applies mutation to offspring individuals in-place.
pub fn mutate(
    individuals: &mut [Individual],
    config: &EngineConfig,
    mutation_type: &MutationType,
    mutation_probability: f64,
    rng: &mut impl Rng,
) {
    let average_fitness = adaptive_average_fitness(individuals, mutation_type);

    for individual in individuals {
        let mut has_mutated = false;

        match adaptive_mutation_num_genes(individual, mutation_type, average_fitness)
            .or(config.mutation_num_genes)
        {
            Some(mutation_num_genes) => {
                has_mutated |= mutate_fixed_num_genes(
                    &mut individual.genes,
                    config,
                    mutation_type,
                    mutation_num_genes,
                    rng,
                );
            }
            None => {
                for (gene_index, gene) in individual.genes.iter_mut().enumerate() {
                    if !rng.gen_bool(mutation_probability) {
                        continue;
                    }

                    has_mutated |= mutate_gene(gene, gene_index, config, mutation_type, rng);
                }
            }
        }

        match mutation_type {
            MutationType::Swap if rng.gen_bool(mutation_probability) => {
                has_mutated |= swap_genes(&mut individual.genes, rng);
            }
            MutationType::Scramble if rng.gen_bool(mutation_probability) => {
                has_mutated |= scramble_genes(&mut individual.genes, rng);
            }
            MutationType::Inversion if rng.gen_bool(mutation_probability) => {
                has_mutated |= invert_genes(&mut individual.genes, rng);
            }
            _ => {}
        }

        if has_mutated {
            individual.clear_evaluation();
        }
    }
}

/// Computes average fitness for adaptive mutation variants.
/// # Panics
fn adaptive_average_fitness(
    individuals: &[Individual],
    mutation_type: &MutationType,
) -> Option<f64> {
    if !is_adaptive_mutation(mutation_type) || individuals.is_empty() {
        return None;
    }

    let total = individuals
        .iter()
        .map(Individual::fitness_or_panic)
        .sum::<f64>();
    Some(total / individuals.len() as f64)
}

/// Returns per-individual mutation gene count for adaptive variants.
/// # Panics
fn adaptive_mutation_num_genes(
    individual: &Individual,
    mutation_type: &MutationType,
    average_fitness: Option<f64>,
) -> Option<usize> {
    let average_fitness = average_fitness?;
    let fitness = individual.fitness_or_panic();

    match mutation_type {
        MutationType::AdaptiveRandomReset {
            low_quality_num_genes,
            high_quality_num_genes,
            ..
        }
        | MutationType::AdaptiveRandomPerturbation {
            low_quality_num_genes,
            high_quality_num_genes,
            ..
        } => Some(if fitness < average_fitness {
            *low_quality_num_genes
        } else {
            *high_quality_num_genes
        }),
        _ => None,
    }
}

/// Returns true when a mutation variant is adaptive.
fn is_adaptive_mutation(mutation_type: &MutationType) -> bool {
    matches!(
        mutation_type,
        MutationType::AdaptiveRandomReset { .. } | MutationType::AdaptiveRandomPerturbation { .. }
    )
}

/// # Returns
fn mutate_fixed_num_genes(
    genes: &mut [crate::ga::core::gene::GeneValue],
    config: &EngineConfig,
    mutation_type: &MutationType,
    mutation_num_genes: usize,
    rng: &mut impl Rng,
) -> bool {
    match mutation_type {
        MutationType::RandomReset { .. }
        | MutationType::RandomPerturbation { .. }
        | MutationType::AdaptiveRandomReset { .. }
        | MutationType::AdaptiveRandomPerturbation { .. } => {}
        MutationType::None
        | MutationType::Swap
        | MutationType::Scramble
        | MutationType::Inversion => return false,
    }

    if mutation_num_genes == 0 || genes.is_empty() {
        return false;
    }

    let mut indices = (0..genes.len()).collect::<Vec<_>>();
    indices.shuffle(rng);

    let mut has_mutated = false;
    for gene_index in indices.into_iter().take(mutation_num_genes) {
        has_mutated |= mutate_gene(
            &mut genes[gene_index],
            gene_index,
            config,
            mutation_type,
            rng,
        );
    }

    has_mutated
}

/// # Returns
fn mutate_gene(
    gene: &mut crate::ga::core::gene::GeneValue,
    gene_index: usize,
    config: &EngineConfig,
    mutation_type: &MutationType,
    rng: &mut impl Rng,
) -> bool {
    match mutation_type {
        MutationType::None => false,
        MutationType::RandomReset { min, max } => {
            *gene = if config.genes_domain.is_some() {
                config.sample_gene(gene_index, rng)
            } else {
                config
                    .normalize_gene(gene_index, rng.gen_range(*min..=*max))
                    .expect("random reset mutation should normalize into a valid value")
            };
            true
        }
        MutationType::AdaptiveRandomReset { min, max, .. } => {
            *gene = if config.genes_domain.is_some() {
                config.sample_gene(gene_index, rng)
            } else {
                config
                    .normalize_gene(gene_index, rng.gen_range(*min..=*max))
                    .expect("adaptive random reset mutation should normalize into a valid value")
            };
            true
        }
        MutationType::RandomPerturbation {
            min_delta,
            max_delta,
        } => {
            let mutated = gene.to_f64() + rng.gen_range(*min_delta..=*max_delta);
            *gene = config
                .normalize_gene(gene_index, mutated)
                .expect("random perturbation mutation should normalize into a valid value");
            true
        }
        MutationType::AdaptiveRandomPerturbation {
            min_delta,
            max_delta,
            ..
        } => {
            let mutated = gene.to_f64() + rng.gen_range(*min_delta..=*max_delta);
            *gene = config.normalize_gene(gene_index, mutated).expect(
                "adaptive random perturbation mutation should normalize into a valid value",
            );
            true
        }
        MutationType::Swap | MutationType::Scramble | MutationType::Inversion => false,
    }
}

/// # Returns
fn swap_genes(genes: &mut [crate::ga::core::gene::GeneValue], rng: &mut impl Rng) -> bool {
    if genes.len() < 2 {
        return false;
    }

    let left = rng.gen_range(0..genes.len());
    let mut right = rng.gen_range(0..genes.len() - 1);
    if right >= left {
        right += 1;
    }

    genes.swap(left, right);
    true
}

/// # Returns
fn scramble_genes(genes: &mut [crate::ga::core::gene::GeneValue], rng: &mut impl Rng) -> bool {
    let Some((start, end)) = random_subsequence_bounds(genes.len(), rng) else {
        return false;
    };

    genes[start..=end].shuffle(rng);
    true
}

/// # Returns
fn invert_genes(genes: &mut [crate::ga::core::gene::GeneValue], rng: &mut impl Rng) -> bool {
    let Some((start, end)) = random_subsequence_bounds(genes.len(), rng) else {
        return false;
    };

    genes[start..=end].reverse();
    true
}

/// - `start < end`
/// - `end < len`
/// # Returns
fn random_subsequence_bounds(len: usize, rng: &mut impl Rng) -> Option<(usize, usize)> {
    if len < 2 {
        return None;
    }

    let start = rng.gen_range(0..len - 1);
    let end = rng.gen_range(start + 1..len);
    Some((start, end))
}
