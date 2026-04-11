use genetic_algorithm_rust::{
    CrossoverType, EngineRunResult, EvolutionEngine, GaConfig, GeneScalarType, GeneValue,
    GenesValueType, MigrationTopology, MutationType, SelectionType, StopCondition,
};

fn fitness(genes: &[GeneValue]) -> f64 {
    genes.iter().map(GeneValue::to_f64).sum()
}

#[test]
fn engine_rejects_single_island_mode() {
    let config = GaConfig::builder(20, 4, 8, 6)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .crossover(CrossoverType::SinglePoint, 0.8)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -0.2,
                max_delta: 0.2,
            },
            0.1,
        )
        .selection_type(SelectionType::Tournament { k: 3 })
        .stop_condition(StopCondition::MaxGenerations)
        .island_model(1, 1, 3, MigrationTopology::Ring)
        .random_seed(Some(11))
        .build();

    assert!(config.is_err());
}

#[test]
fn engine_dispatches_multi_island_mode() {
    let config = GaConfig::builder(20, 4, 8, 6)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .crossover(CrossoverType::SinglePoint, 0.8)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -0.2,
                max_delta: 0.2,
            },
            0.1,
        )
        .selection_type(SelectionType::Tournament { k: 3 })
        .stop_condition(StopCondition::MaxGenerations)
        .island_model(3, 2, 3, MigrationTopology::Ring)
        .random_seed(Some(22))
        .build()
        .unwrap();

    let mut engine = EvolutionEngine::new(config, fitness).unwrap();
    let result = engine.run().unwrap();

    match result {
        EngineRunResult::Island(stats) => {
            assert_eq!(stats.len(), 3);
            assert!(stats.iter().all(|s| !s.best_fitness_per_generation.is_empty()));
            assert!(engine.best_fitness().is_some());
        }
        EngineRunResult::Single(_) => panic!("expected island result"),
    }
}
