use genetic_algorithm_rust::{
    CrossoverType, GaConfig, GeneScalarType, GeneValue, GenesDomain, GenesValueType,
    GeneticAlgorithm, MutationType, SelectionType, StopCondition,
};

fn main() {
    let config = GaConfig::builder(100, 8, 100, 12)
        .init_range(-4.0, 4.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F32))
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
        .expect("invalid GA config");

    let mut ga = GeneticAlgorithm::new(config, |genes| {
        let distance_penalty = genes
            .iter()
            .map(|gene| (gene.to_f64() - 2.0).powi(2))
            .sum::<f64>();
        100.0 - distance_penalty
    })
    .expect("failed to initialize GA");

    ga.run().expect("failed to run GA");
    ga.stats
        .render_report("output/tournament-f32-seed42")
        .expect("failed to render report");

    let best = ga.best_solution().expect("best solution missing");
    println!(
        "Best fitness: {:.4}",
        best.fitness.expect("fitness missing")
    );
    println!(
        "Best genes: {:?}",
        best.genes.iter().map(GeneValue::to_f64).collect::<Vec<_>>()
    );
    println!("Report written to output/tournament-f32-seed42");

    //////////////////////////////////////////////////////////////////////////////////////////////

    let config = GaConfig::builder(100, 8, 200, 12)
        .init_range(-100.0, 100.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::I32))
        .genes_domain(Some(GenesDomain::Global(
            genetic_algorithm_rust::GeneDomain::Continuous {
                low: -100.0,
                high: 100.0,
            },
        )))
        .crossover(CrossoverType::SinglePoint, 0.9)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -1.0,
                max_delta: 1.0,
            },
            0.05,
        )
        .elitism_count(2)
        .selection_type(SelectionType::Rank)
        .random_seed(Some(42))
        .stop_condition(StopCondition::MaxGenerations)
        .build()
        .expect("invalid GA config");

    let mut ga = GeneticAlgorithm::new(config, |genes| {
        let distance_penalty = genes
            .iter()
            .map(|gene| (gene.to_f64() - 2.0).powi(2))
            .sum::<f64>();
        100.0 - distance_penalty
    })
    .expect("failed to initialize GA");

    ga.run().expect("failed to run GA");
    ga.stats
        .render_report("output/rank-i32-seed42")
        .expect("failed to render report");

    let best = ga.best_solution().expect("best solution missing");
    println!(
        "Best fitness: {:.4}",
        best.fitness.expect("fitness missing")
    );
    println!("Best genes: {:?}", best.genes.iter().collect::<Vec<_>>());
    println!("Report written to output/rank-i32-seed42");
}
