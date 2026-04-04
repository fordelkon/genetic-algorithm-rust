use genetic_algorithm_rust::{
    CrossoverType, GaConfig, GeneScalarType, GeneValue, GenesValueType, GeneticAlgorithm,
    MutationType, SelectionType, StopCondition,
};

fn config(seed: u64) -> GaConfig {
    GaConfig::builder(16, 5, 8, 6)
        .init_range(-3.0, 3.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .crossover(CrossoverType::TwoPoint, 0.8)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -0.3,
                max_delta: 0.3,
            },
            0.25,
        )
        .elitism_count(2)
        .selection_type(SelectionType::Tournament { k: 3 })
        .random_seed(Some(seed))
        .stop_condition(StopCondition::MaxGenerations)
        .build()
        .unwrap()
}

#[test]
fn fixed_seed_produces_same_result() {
    let fitness = |genes: &[GeneValue]| {
        -genes
            .iter()
            .map(|gene| (gene.to_f64() - 1.0).powi(2))
            .sum::<f64>()
    };

    let mut ga1 = GeneticAlgorithm::new(config(11), fitness).unwrap();
    let mut ga2 = GeneticAlgorithm::new(config(11), fitness).unwrap();

    ga1.run().unwrap();
    ga2.run().unwrap();

    assert_eq!(ga1.stats.best_solution, ga2.stats.best_solution);
    assert_eq!(ga1.stats.best_fitness, ga2.stats.best_fitness);
}
