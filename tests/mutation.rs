use genetic_algorithm_rust::ga::core::individual::Individual;
use genetic_algorithm_rust::ga::operators::mutation;
use genetic_algorithm_rust::{
    CrossoverType, EngineConfig, GeneDomain, GeneScalarType, GeneValue, GenesDomain,
    GenesValueType, MutationType, SelectionType, StopCondition,
};
use rand::{SeedableRng, rngs::StdRng};

fn sort_genes(genes: &mut [GeneValue]) {
    genes.sort_by(|left, right| {
        left.to_f64()
            .partial_cmp(&right.to_f64())
            .expect("gene comparison should be valid")
    });
}

fn float_config() -> EngineConfig {
    EngineConfig::builder(4, 3, 3, 2)
        .init_range(0.0, 5.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .crossover(CrossoverType::None, 0.0)
        .mutation(MutationType::None, 0.0)
        .elitism_count(1)
        .selection_type(SelectionType::Tournament { k: 2 })
        .random_seed(Some(1))
        .stop_condition(StopCondition::MaxGenerations)
        .build()
        .unwrap()
}

#[test]
fn zero_mutation_probability_keeps_genes() {
    let mut individuals = vec![Individual::new(vec![
        GeneValue::F64(1.0),
        GeneValue::F64(2.0),
        GeneValue::F64(3.0),
    ])];
    let original = individuals[0].genes.clone();
    let mut rng = StdRng::seed_from_u64(5);
    let config = float_config();

    mutation::mutate(
        &mut individuals,
        &config,
        &MutationType::RandomPerturbation {
            min_delta: -1.0,
            max_delta: 1.0,
        },
        0.0,
        &mut rng,
    );

    assert_eq!(individuals[0].genes, original);
}

#[test]
fn random_reset_changes_values_into_range() {
    let mut individuals = vec![Individual::new(vec![
        GeneValue::F64(1.0),
        GeneValue::F64(2.0),
        GeneValue::F64(3.0),
    ])];
    let mut rng = StdRng::seed_from_u64(6);
    let config = float_config();

    mutation::mutate(
        &mut individuals,
        &config,
        &MutationType::RandomReset {
            min: 10.0,
            max: 20.0,
        },
        1.0,
        &mut rng,
    );

    assert!(
        individuals[0]
            .genes
            .iter()
            .all(|gene| (10.0..=20.0).contains(&gene.to_f64()))
    );
}

#[test]
fn integer_gene_type_keeps_integer_values_after_mutation() {
    let mut individuals = vec![Individual::new(vec![
        GeneValue::I32(1),
        GeneValue::I32(2),
        GeneValue::I32(3),
    ])];
    let mut rng = StdRng::seed_from_u64(9);
    let config = EngineConfig::builder(4, 3, 3, 2)
        .init_range(0.0, 5.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::I32))
        .genes_domain(Some(GenesDomain::Global(GeneDomain::Continuous {
            low: 0.0,
            high: 10.0,
        })))
        .crossover(CrossoverType::None, 0.0)
        .mutation(MutationType::None, 0.0)
        .elitism_count(1)
        .selection_type(SelectionType::Tournament { k: 2 })
        .random_seed(Some(1))
        .stop_condition(StopCondition::MaxGenerations)
        .build()
        .unwrap();

    mutation::mutate(
        &mut individuals,
        &config,
        &MutationType::RandomPerturbation {
            min_delta: -0.75,
            max_delta: 0.75,
        },
        1.0,
        &mut rng,
    );

    assert!(
        individuals[0]
            .genes
            .iter()
            .all(|gene| matches!(gene, GeneValue::I32(_)) && (0.0..=10.0).contains(&gene.to_f64()))
    );
}

#[test]
fn random_reset_uses_gene_space_when_present() {
    let mut individuals = vec![Individual::new(vec![
        GeneValue::F64(1.0),
        GeneValue::F64(2.0),
        GeneValue::F64(3.0),
    ])];
    let mut rng = StdRng::seed_from_u64(10);
    let config = EngineConfig::builder(4, 3, 3, 2)
        .init_range(0.0, 5.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .genes_domain(Some(GenesDomain::PerGene(vec![
            GeneDomain::Discrete(vec![100.0]),
            GeneDomain::Discrete(vec![200.0, 300.0]),
            GeneDomain::Stepped {
                low: 10.0,
                high: 12.0,
                step: 1.0,
            },
        ])))
        .crossover(CrossoverType::None, 0.0)
        .mutation(MutationType::None, 0.0)
        .elitism_count(1)
        .selection_type(SelectionType::Tournament { k: 2 })
        .random_seed(Some(1))
        .stop_condition(StopCondition::MaxGenerations)
        .build()
        .unwrap();

    mutation::mutate(
        &mut individuals,
        &config,
        &MutationType::RandomReset {
            min: -999.0,
            max: -100.0,
        },
        1.0,
        &mut rng,
    );

    assert_eq!(individuals[0].genes[0], GeneValue::F64(100.0));
    assert!(matches!(
        individuals[0].genes[1],
        GeneValue::F64(200.0) | GeneValue::F64(300.0)
    ));
    assert!(matches!(
        individuals[0].genes[2],
        GeneValue::F64(10.0) | GeneValue::F64(11.0) | GeneValue::F64(12.0)
    ));
}

#[test]
fn swap_mutation_reorders_genes_and_clears_fitness() {
    let original = vec![
        GeneValue::I32(1),
        GeneValue::I32(2),
        GeneValue::I32(3),
        GeneValue::I32(4),
    ];
    let mut individuals = vec![Individual::with_fitness(original.clone(), 42.0)];
    let mut rng = StdRng::seed_from_u64(12);
    let config = EngineConfig::builder(4, 4, 3, 2)
        .genes_value_type(GenesValueType::All(GeneScalarType::I32))
        .crossover(CrossoverType::None, 0.0)
        .mutation(MutationType::None, 0.0)
        .build()
        .unwrap();

    mutation::mutate(
        &mut individuals,
        &config,
        &MutationType::Swap,
        1.0,
        &mut rng,
    );

    assert_ne!(individuals[0].genes, original);
    let mut mutated = individuals[0].genes.clone();
    sort_genes(&mut mutated);
    let mut expected = original.clone();
    sort_genes(&mut expected);
    assert_eq!(mutated, expected);
    assert!(individuals[0].evaluation.is_none());
}

#[test]
fn scramble_mutation_scrambles_subsequence_without_changing_members() {
    let original = vec![
        GeneValue::I32(1),
        GeneValue::I32(2),
        GeneValue::I32(3),
        GeneValue::I32(4),
        GeneValue::I32(5),
        GeneValue::I32(6),
    ];
    let mut individuals = vec![Individual::with_fitness(original.clone(), 42.0)];
    let mut rng = StdRng::seed_from_u64(13);
    let config = EngineConfig::builder(4, 6, 3, 2)
        .genes_value_type(GenesValueType::All(GeneScalarType::I32))
        .crossover(CrossoverType::None, 0.0)
        .mutation(MutationType::None, 0.0)
        .build()
        .unwrap();

    mutation::mutate(
        &mut individuals,
        &config,
        &MutationType::Scramble,
        1.0,
        &mut rng,
    );

    assert_ne!(individuals[0].genes, original);
    let mut mutated = individuals[0].genes.clone();
    sort_genes(&mut mutated);
    let mut expected = original.clone();
    sort_genes(&mut expected);
    assert_eq!(mutated, expected);
    assert!(individuals[0].evaluation.is_none());
}

#[test]
fn inversion_mutation_reverses_subsequence_without_changing_members() {
    let original = vec![
        GeneValue::I32(1),
        GeneValue::I32(2),
        GeneValue::I32(3),
        GeneValue::I32(4),
        GeneValue::I32(5),
        GeneValue::I32(6),
    ];
    let mut individuals = vec![Individual::with_fitness(original.clone(), 42.0)];
    let mut rng = StdRng::seed_from_u64(14);
    let config = EngineConfig::builder(4, 6, 3, 2)
        .genes_value_type(GenesValueType::All(GeneScalarType::I32))
        .crossover(CrossoverType::None, 0.0)
        .mutation(MutationType::None, 0.0)
        .build()
        .unwrap();

    mutation::mutate(
        &mut individuals,
        &config,
        &MutationType::Inversion,
        1.0,
        &mut rng,
    );

    assert_ne!(individuals[0].genes, original);
    let mut mutated = individuals[0].genes.clone();
    sort_genes(&mut mutated);
    let mut expected = original.clone();
    sort_genes(&mut expected);
    assert_eq!(mutated, expected);
    assert!(individuals[0].evaluation.is_none());
}

#[test]
fn mutation_num_genes_mutates_exactly_k_genes_for_random_reset() {
    let original = vec![
        GeneValue::F64(1.0),
        GeneValue::F64(2.0),
        GeneValue::F64(3.0),
        GeneValue::F64(4.0),
        GeneValue::F64(5.0),
    ];
    let mut individuals = vec![Individual::new(original.clone())];
    let mut rng = StdRng::seed_from_u64(15);
    let config = EngineConfig::builder(4, 5, 3, 2)
        .init_range(0.0, 5.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .crossover(CrossoverType::None, 0.0)
        .mutation(MutationType::None, 0.0)
        .mutation_num_genes(2)
        .build()
        .unwrap();

    mutation::mutate(
        &mut individuals,
        &config,
        &MutationType::RandomReset {
            min: 100.0,
            max: 200.0,
        },
        0.0,
        &mut rng,
    );

    let changed = individuals[0]
        .genes
        .iter()
        .zip(original.iter())
        .filter(|(left, right)| left != right)
        .count();
    assert_eq!(changed, 2);
}

#[test]
fn mutation_num_genes_is_rejected_when_exceeding_num_genes() {
    let error = EngineConfig::builder(4, 3, 3, 2)
        .mutation_num_genes(4)
        .build()
        .unwrap_err();

    assert!(matches!(
        error,
        genetic_algorithm_rust::GaError::InvalidConfig(_)
    ));
}

#[test]
fn adaptive_random_reset_mutates_more_genes_for_low_fitness_individuals() {
    let low_original = vec![
        GeneValue::F64(1.0),
        GeneValue::F64(2.0),
        GeneValue::F64(3.0),
        GeneValue::F64(4.0),
    ];
    let high_original = vec![
        GeneValue::F64(10.0),
        GeneValue::F64(20.0),
        GeneValue::F64(30.0),
        GeneValue::F64(40.0),
    ];
    let mut individuals = vec![
        Individual::with_fitness(low_original.clone(), 1.0),
        Individual::with_fitness(high_original.clone(), 10.0),
    ];
    let mut rng = StdRng::seed_from_u64(16);
    let config = EngineConfig::builder(4, 4, 3, 2)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .crossover(CrossoverType::None, 0.0)
        .mutation(MutationType::None, 0.0)
        .build()
        .unwrap();

    mutation::mutate(
        &mut individuals,
        &config,
        &MutationType::AdaptiveRandomReset {
            min: 100.0,
            max: 200.0,
            low_quality_num_genes: 3,
            high_quality_num_genes: 1,
        },
        0.0,
        &mut rng,
    );

    let low_changed = individuals[0]
        .genes
        .iter()
        .zip(low_original.iter())
        .filter(|(left, right)| left != right)
        .count();
    let high_changed = individuals[1]
        .genes
        .iter()
        .zip(high_original.iter())
        .filter(|(left, right)| left != right)
        .count();

    assert_eq!(low_changed, 3);
    assert_eq!(high_changed, 1);
    assert!(
        individuals
            .iter()
            .all(|individual| individual.evaluation.is_none())
    );
}

#[test]
fn adaptive_mutation_gene_counts_are_rejected_when_exceeding_num_genes() {
    let error = EngineConfig::builder(4, 3, 3, 2)
        .mutation(
            MutationType::AdaptiveRandomReset {
                min: 0.0,
                max: 1.0,
                low_quality_num_genes: 4,
                high_quality_num_genes: 1,
            },
            0.0,
        )
        .build()
        .unwrap_err();

    assert!(matches!(
        error,
        genetic_algorithm_rust::GaError::InvalidConfig(_)
    ));
}
