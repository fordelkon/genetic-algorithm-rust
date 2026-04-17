use genetic_algorithm_rust::{
    CrossoverType, EngineConfig, EvolutionEngine, GeneDomain, GeneScalarType, GeneValue,
    GenesDomain, GenesValueType, MigrationType, MutationType, OptimizationMode, StopCondition,
};

fn nsga2_config(num_islands: usize) -> EngineConfig {
    EngineConfig::builder(24, 2, 12, 8)
        .init_range(-2.0, 2.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .genes_domain(Some(GenesDomain::Global(GeneDomain::Continuous {
            low: -2.0,
            high: 2.0,
        })))
        .crossover(CrossoverType::SinglePoint, 0.85)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -0.15,
                max_delta: 0.15,
            },
            0.2,
        )
        .optimization_mode(OptimizationMode::Nsga2 { num_objectives: 2 })
        .stop_condition(StopCondition::MaxGenerations)
        .random_seed(Some(19))
        .island_model(num_islands, 2, 4, MigrationType::Ring)
        .build()
        .unwrap()
}

fn dominates(left: &[f64], right: &[f64]) -> bool {
    left.iter().zip(right.iter()).all(|(l, r)| l <= r)
        && left.iter().zip(right.iter()).any(|(l, r)| l < r)
}

#[test]
fn nsga2_rejects_target_fitness_stop_condition() {
    let result = EngineConfig::builder(20, 2, 10, 6)
        .optimization_mode(OptimizationMode::Nsga2 { num_objectives: 2 })
        .stop_condition(StopCondition::TargetFitness(0.0))
        .build();

    assert!(result.is_err());
}

#[test]
fn nsga2_exposes_a_non_dominated_pareto_front() {
    let config = nsga2_config(1);
    let mut engine = EvolutionEngine::new(config, |genes: &[GeneValue]| {
        let x = genes[0].to_f64();
        let y = genes[1].to_f64();
        vec![x.powi(2) + y.powi(2), (x - 1.0).powi(2) + (y + 1.0).powi(2)]
    })
    .unwrap();

    engine.run().unwrap();
    let front = engine.pareto_front().unwrap();

    assert!(!front.is_empty());
    assert!(front.iter().all(|solution| solution.rank == 0));

    for (i, left) in front.iter().enumerate() {
        for (j, right) in front.iter().enumerate() {
            if i == j {
                continue;
            }

            assert!(!dominates(&left.objectives, &right.objectives));
        }
    }
}

#[test]
fn nsga2_island_run_renders_multi_objective_report() {
    let config = nsga2_config(3);
    let mut engine = EvolutionEngine::new(config, |genes: &[GeneValue]| {
        let x = genes[0].to_f64();
        let y = genes[1].to_f64();
        vec![x.powi(2) + y.powi(2), (x - 1.5).powi(2) + (y - 0.5).powi(2)]
    })
    .unwrap();

    engine.run().unwrap();

    let output_dir = std::env::temp_dir().join("ga-nsga2-report");
    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir).unwrap();
    }

    engine.stats.render_report(&output_dir).unwrap();

    assert!(output_dir.join("front_size_history.svg").exists());
    assert!(output_dir.join("pareto_front.svg").exists());
    assert!(output_dir.join("summary.md").exists());
}
