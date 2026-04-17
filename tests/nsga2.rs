use genetic_algorithm_rust::{
    CrossoverType, EngineConfig, EvolutionEngine, GeneDomain, GeneScalarType, GeneValue,
    GenesDomain, GenesValueType, MigrationType, MutationType, OptimizationMode, StopCondition,
};

fn nsga2_config(num_islands: usize) -> EngineConfig {
    EngineConfig::builder(24, 2, 12, 8)
        .init_range(0.0, 1.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .genes_domain(Some(GenesDomain::Global(GeneDomain::Continuous {
            low: 0.0,
            high: 1.0,
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

fn zdt2_2d(genes: &[GeneValue]) -> Vec<f64> {
    let x1 = genes[0].to_f64();
    let x2 = genes[1].to_f64();
    let f1 = x1;
    let g = 1.0 + 9.0 * x2;
    let f2 = g * (1.0 - (f1 / g).powi(2));
    vec![f1, f2]
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
fn zdt2_2d_matches_reference_formula() {
    let objectives = zdt2_2d(&[GeneValue::F64(0.5), GeneValue::F64(0.25)]);

    assert!((objectives[0] - 0.5).abs() < 1e-12);
    assert!((objectives[1] - 3.1730769230769234).abs() < 1e-12);
}

#[test]
fn nsga2_exposes_a_non_dominated_pareto_front() {
    let config = nsga2_config(1);
    let mut engine = EvolutionEngine::new(config, zdt2_2d).unwrap();

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
    let mut engine = EvolutionEngine::new(config, zdt2_2d).unwrap();

    engine.run().unwrap();

    let output_dir = std::env::temp_dir().join("ga-nsga2-report");
    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir).unwrap();
    }

    engine.stats.render_report(&output_dir).unwrap();

    assert!(output_dir.join("front_size_history.svg").exists());
    assert!(output_dir.join("pareto_front.svg").exists());
    assert!(output_dir.join("pareto_priority.svg").exists());
    assert!(output_dir.join("summary.md").exists());

    let front_size_svg = std::fs::read_to_string(output_dir.join("front_size_history.svg")).unwrap();
    assert!(
        front_size_svg.contains("<path")
            || front_size_svg.contains("<polyline")
            || front_size_svg.contains("<line"),
        "front size history should contain actual chart primitives"
    );
    assert!(front_size_svg.contains("Front 0 Size"));
    assert!(front_size_svg.contains("Front Count"));

    let pareto_svg = std::fs::read_to_string(output_dir.join("pareto_front.svg")).unwrap();
    assert!(
        pareto_svg.contains("<circle") || pareto_svg.contains("<path"),
        "pareto front should contain plotted points"
    );
    assert!(pareto_svg.contains("Rank"));
    assert!(pareto_svg.contains("Crowding"));

    let priority_svg = std::fs::read_to_string(output_dir.join("pareto_priority.svg")).unwrap();
    assert!(
        priority_svg.contains("<rect") || priority_svg.contains("<path"),
        "priority chart should contain ranked glyphs"
    );
    assert!(priority_svg.contains("Crowding Distance"));
    assert!(priority_svg.contains("NSGA-II Priority"));
}
