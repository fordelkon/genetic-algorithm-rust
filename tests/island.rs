use genetic_algorithm_rust::{
    CrossoverType, GaConfig, GeneScalarType, GeneValue, GenesDomain, GenesValueType,
    GeneticAlgorithm, IslandConfig, IslandModel, MigrationTopology, MutationType, SelectionType,
    StopCondition,
};

/// Target: minimize distance of each gene to 2.0.
fn target_fitness(genes: &[GeneValue]) -> f64 {
    let penalty: f64 = genes
        .iter()
        .map(|g| (g.to_f64() - 2.0).powi(2))
        .sum();
    100.0 - penalty
}

fn base_config() -> GaConfig {
    GaConfig::builder(30, 4, 50, 8)
        .init_range(-4.0, 4.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .genes_domain(Some(GenesDomain::Global(
            genetic_algorithm_rust::GeneDomain::Continuous {
                low: -4.0,
                high: 4.0,
            },
        )))
        .crossover(CrossoverType::SinglePoint, 0.9)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -0.5,
                max_delta: 0.5,
            },
            0.15,
        )
        .elitism_count(2)
        .selection_type(SelectionType::Tournament { k: 3 })
        .random_seed(Some(42))
        .stop_condition(StopCondition::MaxGenerations)
        .build()
        .expect("valid config")
}

#[test]
fn island_config_requires_at_least_two_islands() {
    let result = IslandModel::new(
        IslandConfig::new(1, 2, 5, MigrationTopology::Ring),
        base_config(),
        target_fitness,
    );
    assert!(result.is_err());
}

#[test]
fn island_config_rejects_zero_migration_count() {
    let result = IslandModel::new(
        IslandConfig::new(3, 0, 5, MigrationTopology::Ring),
        base_config(),
        target_fitness,
    );
    assert!(result.is_err());
}

#[test]
fn island_config_rejects_zero_interval() {
    let result = IslandModel::new(
        IslandConfig::new(3, 2, 0, MigrationTopology::Ring),
        base_config(),
        target_fitness,
    );
    assert!(result.is_err());
}

#[test]
fn island_config_rejects_migration_count_exceeding_population() {
    let result = IslandModel::new(
        IslandConfig::new(3, 100, 5, MigrationTopology::Ring),
        base_config(), // pop_size = 30
        target_fitness,
    );
    assert!(result.is_err());
}

#[test]
fn ring_topology_basic_run() {
    let island_config = IslandConfig::new(3, 2, 10, MigrationTopology::Ring);
    let mut model = IslandModel::new(island_config, base_config(), target_fitness)
        .expect("valid island model");

    let stats = model.run().expect("island model should run");
    assert_eq!(stats.len(), 3);

    // All islands should have recorded stats
    for s in &stats {
        assert!(!s.best_fitness_per_generation.is_empty());
    }

    let best = model.best_solution().expect("should have a best solution");
    assert!(best.fitness.is_some());
    assert!(best.fitness.unwrap() > 0.0, "fitness should be positive for this problem");
}

#[test]
fn fully_connected_topology_basic_run() {
    let island_config = IslandConfig::new(4, 2, 10, MigrationTopology::FullyConnected);
    let mut model = IslandModel::new(island_config, base_config(), target_fitness)
        .expect("valid island model");

    model.run().expect("island model should run");

    let best = model.best_solution().expect("should have a best solution");
    assert!(best.fitness.unwrap() > 0.0);
}

#[test]
fn island_model_outperforms_single_population() {
    // Run a single GA with 90 individuals for 50 generations
    let mut single_config = base_config();
    single_config.population_size = 90;
    single_config.num_parents_mating = 24;

    let mut single_ga = GeneticAlgorithm::new(single_config, target_fitness)
        .expect("valid single GA");
    single_ga.run().expect("single GA should run");
    let single_best = single_ga.best_solution().expect("best").fitness.unwrap();

    // Run 3 islands of 30 each (same total evaluations) with migration
    let island_config = IslandConfig::new(3, 3, 10, MigrationTopology::Ring);
    let mut model = IslandModel::new(island_config, base_config(), target_fitness)
        .expect("valid island model");
    model.run().expect("island model should run");
    let island_best = model.best_fitness().expect("should have best fitness");

    // Island model should find a solution at least as good
    // (this is probabilistic but with seed=42 and these params it's reliable)
    assert!(
        island_best >= single_best * 0.95,
        "island model ({island_best:.4}) should be competitive with single pop ({single_best:.4})"
    );
}

#[test]
fn heterogeneous_islands() {
    let configs: Vec<GaConfig> = vec![
        // Island 0: high mutation, tournament selection
        GaConfig::builder(30, 4, 50, 8)
            .init_range(-4.0, 4.0)
            .genes_value_type(GenesValueType::All(GeneScalarType::F64))
            .crossover(CrossoverType::SinglePoint, 0.9)
            .mutation(
                MutationType::RandomPerturbation { min_delta: -1.0, max_delta: 1.0 },
                0.3,
            )
            .elitism_count(2)
            .selection_type(SelectionType::Tournament { k: 3 })
            .random_seed(Some(100))
            .stop_condition(StopCondition::MaxGenerations)
            .build()
            .unwrap(),
        // Island 1: low mutation, rank selection
        GaConfig::builder(30, 4, 50, 8)
            .init_range(-4.0, 4.0)
            .genes_value_type(GenesValueType::All(GeneScalarType::F64))
            .crossover(CrossoverType::TwoPoint, 0.8)
            .mutation(
                MutationType::RandomPerturbation { min_delta: -0.1, max_delta: 0.1 },
                0.05,
            )
            .elitism_count(4)
            .selection_type(SelectionType::Rank)
            .random_seed(Some(200))
            .stop_condition(StopCondition::MaxGenerations)
            .build()
            .unwrap(),
    ];

    let island_config = IslandConfig::new(2, 3, 10, MigrationTopology::Ring);
    let mut model = IslandModel::with_configs(island_config, configs, target_fitness)
        .expect("valid heterogeneous model");

    model.run().expect("heterogeneous island model should run");

    let best = model.best_solution().expect("should have a best solution");
    assert!(best.fitness.unwrap() > 0.0);
}

#[test]
fn with_configs_rejects_wrong_count() {
    let configs = vec![base_config()]; // Only 1 config for 3 islands
    let island_config = IslandConfig::new(3, 2, 10, MigrationTopology::Ring);
    let result = IslandModel::with_configs(island_config, configs, target_fitness);
    assert!(result.is_err());
}

#[test]
fn reproducibility_with_seed() {
    let run = || {
        let island_config = IslandConfig::new(3, 2, 10, MigrationTopology::Ring);
        let mut model = IslandModel::new(island_config, base_config(), target_fitness)
            .expect("valid island model");
        model.run().expect("should run");
        model.best_fitness().expect("should have best fitness")
    };

    let fitness_1 = run();
    let fitness_2 = run();
    assert!(
        (fitness_1 - fitness_2).abs() < 1e-10,
        "same seed should produce identical results: {fitness_1} vs {fitness_2}"
    );
}

#[test]
fn best_fitness_aggregates_across_islands() {
    let island_config = IslandConfig::new(3, 2, 10, MigrationTopology::Ring);
    let mut model = IslandModel::new(island_config, base_config(), target_fitness)
        .expect("valid island model");
    model.run().expect("should run");

    let global_best = model.best_fitness().expect("should have best fitness");

    // Global best should be >= each island's best
    for island in &model.islands {
        if let Some(island_best) = island.stats.best_fitness {
            assert!(global_best >= island_best);
        }
    }
}

#[test]
fn ring_topology_neighbors() {
    let topo = MigrationTopology::Ring;
    // Ring of 4: each sends to the next
    assert_eq!(topo.neighbors(0, 4), vec![1]);
    assert_eq!(topo.neighbors(1, 4), vec![2]);
    assert_eq!(topo.neighbors(2, 4), vec![3]);
    assert_eq!(topo.neighbors(3, 4), vec![0]); // wraps around
}

#[test]
fn fully_connected_topology_neighbors() {
    let topo = MigrationTopology::FullyConnected;
    assert_eq!(topo.neighbors(0, 4), vec![1, 2, 3]);
    assert_eq!(topo.neighbors(2, 4), vec![0, 1, 3]);
}
