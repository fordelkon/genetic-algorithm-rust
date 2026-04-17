use genetic_algorithm_rust::{
    CrossoverType, EngineConfig, EngineKernel, GeneDomain, GeneScalarType, GeneValue, GenesDomain,
    GenesValueType, MutationType, SelectionType, StopCondition,
};
use rayon::ThreadPoolBuilder;
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

fn base_config() -> EngineConfig {
    EngineConfig::builder(20, 6, 10, 8)
        .init_range(-2.0, 2.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .genes_domain(Some(GenesDomain::Global(GeneDomain::Continuous {
            low: -2.0,
            high: 2.0,
        })))
        .crossover(CrossoverType::SinglePoint, 0.9)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -0.2,
                max_delta: 0.2,
            },
            0.2,
        )
        .elitism_count(2)
        .selection_type(SelectionType::Tournament { k: 3 })
        .random_seed(Some(7))
        .stop_condition(StopCondition::MaxGenerations)
        .build()
        .unwrap()
}

#[test]
fn ga_runs_and_records_history() {
    let mut ga = EngineKernel::new(base_config(), |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();
    let stats = ga.run().unwrap();

    assert!(!stats.best_fitness_per_generation.is_empty());
    assert!(stats.best_solution.is_some());
    assert!(ga.best_solution().unwrap().evaluation.is_some());
}

#[test]
fn ga_records_best_genes_per_generation() {
    let mut ga = EngineKernel::new(base_config(), |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();

    let stats = ga.run().unwrap();

    assert_eq!(
        stats.best_genes_per_generation.len(),
        stats.best_fitness_per_generation.len()
    );
    assert_eq!(
        stats.best_genes_per_generation[0].len(),
        ga.config.num_genes
    );
}

#[test]
fn ga_records_fitness_standard_deviation_per_generation() {
    let mut ga = EngineKernel::new(base_config(), |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();

    let stats = ga.run().unwrap();

    assert_eq!(
        stats.std_fitness_per_generation.len(),
        stats.avg_fitness_per_generation.len()
    );
    assert!(
        stats
            .std_fitness_per_generation
            .iter()
            .all(|value| *value >= 0.0)
    );
}

#[test]
fn report_summary_exposes_core_metrics() {
    let mut ga = EngineKernel::new(base_config(), |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();
    let stats = ga.run().unwrap();

    let summary = stats.summary().unwrap();

    assert_eq!(summary.generations, stats.best_fitness_per_generation.len());
    assert_eq!(summary.best_fitness, stats.best_fitness.unwrap());
    assert_eq!(
        summary.final_avg_fitness,
        *stats.avg_fitness_per_generation.last().unwrap()
    );
}

#[test]
fn report_summary_includes_final_std_fitness() {
    let mut ga = EngineKernel::new(base_config(), |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();
    let stats = ga.run().unwrap();

    let summary = stats.summary().unwrap();

    assert_eq!(
        summary.final_std_fitness,
        *stats.std_fitness_per_generation.last().unwrap()
    );
}

#[test]
fn render_report_writes_expected_artifacts() {
    let mut ga = EngineKernel::new(base_config(), |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();
    let stats = ga.run().unwrap();

    let output_dir = std::env::temp_dir().join("ga-report-test");
    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir).unwrap();
    }

    stats.render_report(&output_dir).unwrap();

    assert!(output_dir.join("fitness_history.svg").exists());
    assert!(output_dir.join("best_genes_final.svg").exists());
    assert!(output_dir.join("best_genes_trajectory.svg").exists());
    assert!(output_dir.join("summary.md").exists());
}

#[test]
fn render_report_labels_best_gene_axis_with_gene_ids() {
    let mut ga = EngineKernel::new(base_config(), |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();
    let stats = ga.run().unwrap();

    let output_dir = std::env::temp_dir().join("ga-report-gene-axis");
    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir).unwrap();
    }

    stats.render_report(&output_dir).unwrap();
    let svg = std::fs::read_to_string(output_dir.join("best_genes_final.svg")).unwrap();

    assert!(svg.contains("g0"));
    assert!(svg.contains("g1"));
}

#[test]
fn render_report_writes_fitness_history_with_band_and_smoothed_line() {
    let mut ga = EngineKernel::new(base_config(), |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();
    let stats = ga.run().unwrap();

    let output_dir = std::env::temp_dir().join("ga-report-polish");
    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir).unwrap();
    }

    stats.render_report(&output_dir).unwrap();
    let svg = std::fs::read_to_string(output_dir.join("fitness_history.svg")).unwrap();

    assert!(
        svg.contains("<path") || svg.contains("<polyline") || svg.contains("<line"),
        "fitness history should contain actual chart primitives"
    );
    assert!(svg.contains("Average Fitness"));
    assert!(svg.contains("Smoothed Average"));
}

#[test]
fn render_report_writes_single_page_gene_trajectory_with_omission_card() {
    let config = EngineConfig::builder(20, 13, 10, 8)
        .init_range(-2.0, 2.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .selection_type(SelectionType::Tournament { k: 3 })
        .random_seed(Some(9))
        .build()
        .unwrap();
    let mut ga = EngineKernel::new(config, |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();
    let stats = ga.run().unwrap();

    let output_dir = std::env::temp_dir().join("ga-report-pages");
    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir).unwrap();
    }

    stats.render_report(&output_dir).unwrap();

    let trajectory_path = output_dir.join("best_genes_trajectory.svg");
    assert!(trajectory_path.exists());
    assert!(!output_dir.join("best_genes_trajectory_page_1.svg").exists());

    let svg = std::fs::read_to_string(trajectory_path).unwrap();
    assert!(
        svg.contains("<path") || svg.contains("<polyline") || svg.contains("<line"),
        "gene trajectory should contain actual chart primitives"
    );
    assert!(svg.contains("Omitted Genes"));
}

#[test]
fn render_report_elides_long_best_gene_final_chart_with_ellipsis() {
    let config = EngineConfig::builder(20, 13, 10, 8)
        .init_range(-2.0, 2.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .selection_type(SelectionType::Tournament { k: 3 })
        .random_seed(Some(11))
        .build()
        .unwrap();
    let mut ga = EngineKernel::new(config, |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();
    let stats = ga.run().unwrap();

    let output_dir = std::env::temp_dir().join("ga-report-final-ellipsis");
    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir).unwrap();
    }

    stats.render_report(&output_dir).unwrap();

    let svg = std::fs::read_to_string(output_dir.join("best_genes_final.svg")).unwrap();
    assert!(svg.contains("..."));
    assert!(svg.contains("g0"));
    assert!(svg.contains("g3"));
    assert!(svg.contains("g9"));
    assert!(svg.contains("g12"));
    assert!(!svg.contains("g4"));
}

#[test]
fn render_report_marks_omitted_long_gene_trajectory() {
    let config = EngineConfig::builder(20, 13, 10, 8)
        .init_range(-2.0, 2.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .selection_type(SelectionType::Tournament { k: 3 })
        .random_seed(Some(12))
        .build()
        .unwrap();
    let mut ga = EngineKernel::new(config, |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();
    let stats = ga.run().unwrap();

    let output_dir = std::env::temp_dir().join("ga-report-trajectory-ellipsis");
    if output_dir.exists() {
        std::fs::remove_dir_all(&output_dir).unwrap();
    }

    stats.render_report(&output_dir).unwrap();

    let svg = std::fs::read_to_string(output_dir.join("best_genes_trajectory.svg")).unwrap();
    assert!(svg.contains("Omitted Genes"));
    assert!(svg.contains("5"));
}

#[test]
fn initialization_respects_discrete_genes_domain() {
    let config = EngineConfig::builder(20, 6, 10, 8)
        .init_range(-2.0, 2.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .genes_domain(Some(GenesDomain::PerGene(vec![
            GeneDomain::Discrete(vec![1.0, 3.0]),
            GeneDomain::Discrete(vec![2.0, 4.0]),
            GeneDomain::Stepped {
                low: 10.0,
                high: 14.0,
                step: 2.0,
            },
            GeneDomain::Continuous {
                low: -1.0,
                high: 1.0,
            },
            GeneDomain::Discrete(vec![8.0]),
            GeneDomain::Discrete(vec![5.0, 6.0, 7.0]),
        ])))
        .crossover(CrossoverType::SinglePoint, 0.9)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -0.2,
                max_delta: 0.2,
            },
            0.2,
        )
        .elitism_count(2)
        .selection_type(SelectionType::Tournament { k: 3 })
        .random_seed(Some(7))
        .stop_condition(StopCondition::MaxGenerations)
        .build()
        .unwrap();

    let mut ga = EngineKernel::new(config, |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();
    ga.initialize_population().unwrap();

    for individual in &ga.population.individuals {
        assert!(matches!(
            individual.genes[0],
            GeneValue::F64(1.0) | GeneValue::F64(3.0)
        ));
        assert!(matches!(
            individual.genes[1],
            GeneValue::F64(2.0) | GeneValue::F64(4.0)
        ));
        assert!(matches!(
            individual.genes[2],
            GeneValue::F64(10.0) | GeneValue::F64(12.0) | GeneValue::F64(14.0)
        ));
        assert!((-1.0..=1.0).contains(&individual.genes[3].to_f64()));
        assert_eq!(individual.genes[4], GeneValue::F64(8.0));
        assert!(matches!(
            individual.genes[5],
            GeneValue::F64(5.0) | GeneValue::F64(6.0) | GeneValue::F64(7.0)
        ));
    }
}

#[test]
fn mixed_scalar_types_run_with_real_variants() {
    let config = EngineConfig::builder(12, 4, 6, 4)
        .genes_value_type(GenesValueType::PerGene(vec![
            GeneScalarType::I8,
            GeneScalarType::U16,
            GeneScalarType::F32,
            GeneScalarType::F64,
        ]))
        .genes_domain(Some(GenesDomain::PerGene(vec![
            GeneDomain::Discrete(vec![-3.0, -1.0, 2.0]),
            GeneDomain::Stepped {
                low: 0.0,
                high: 20.0,
                step: 5.0,
            },
            GeneDomain::Continuous {
                low: -1.0,
                high: 1.0,
            },
            GeneDomain::Continuous {
                low: -10.0,
                high: 10.0,
            },
        ])))
        .selection_type(SelectionType::Tournament { k: 3 })
        .random_seed(Some(21))
        .build()
        .unwrap();

    let mut ga = EngineKernel::new(config, |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();
    ga.run().unwrap();

    let best = ga.best_solution().unwrap();
    assert!(matches!(best.genes[0], GeneValue::I8(_)));
    assert!(matches!(best.genes[1], GeneValue::U16(_)));
    assert!(matches!(best.genes[2], GeneValue::F32(_)));
    assert!(matches!(best.genes[3], GeneValue::F64(_)));
}

#[test]
fn unsupported_float16_gene_type_is_rejected() {
    let error = EngineConfig::builder(8, 2, 4, 2)
        .genes_value_type(GenesValueType::All(GeneScalarType::Float16))
        .build()
        .unwrap_err();

    assert!(matches!(
        error,
        genetic_algorithm_rust::GaError::UnsupportedGeneType(_)
    ));
}

#[test]
fn unsigned_domain_with_negative_values_is_rejected() {
    let error = EngineConfig::builder(8, 2, 4, 2)
        .genes_value_type(GenesValueType::All(GeneScalarType::U8))
        .genes_domain(Some(GenesDomain::Global(GeneDomain::Continuous {
            low: -1.0,
            high: 10.0,
        })))
        .build()
        .unwrap_err();

    assert!(matches!(
        error,
        genetic_algorithm_rust::GaError::InvalidConfig(_)
    ));
}

#[test]
fn ga_runs_with_adaptive_mutation() {
    let config = EngineConfig::builder(20, 6, 8, 8)
        .init_range(-2.0, 2.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .crossover(CrossoverType::SinglePoint, 0.9)
        .mutation(
            MutationType::AdaptiveRandomPerturbation {
                min_delta: -0.2,
                max_delta: 0.2,
                low_quality_num_genes: 3,
                high_quality_num_genes: 1,
            },
            1.0,
        )
        .elitism_count(2)
        .selection_type(SelectionType::Tournament { k: 3 })
        .random_seed(Some(7))
        .stop_condition(StopCondition::MaxGenerations)
        .build()
        .unwrap();

    let mut ga = EngineKernel::new(config, |genes| {
        genes.iter().map(GeneValue::to_f64).sum::<f64>()
    })
    .unwrap();

    let stats = ga.run().unwrap();

    assert!(!stats.best_fitness_per_generation.is_empty());
    assert!(ga.best_solution().unwrap().evaluation.is_some());
}

#[test]
fn evaluate_population_runs_fitness_in_parallel() {
    let config = EngineConfig::builder(64, 4, 1, 2)
        .init_range(-2.0, 2.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .selection_type(SelectionType::Tournament { k: 2 })
        .random_seed(Some(7))
        .build()
        .unwrap();

    let in_flight = Arc::new(AtomicUsize::new(0));
    let max_in_flight = Arc::new(AtomicUsize::new(0));

    let fitness = {
        let in_flight = Arc::clone(&in_flight);
        let max_in_flight = Arc::clone(&max_in_flight);
        move |genes: &[GeneValue]| {
            let current = in_flight.fetch_add(1, Ordering::SeqCst) + 1;

            let mut observed_max = max_in_flight.load(Ordering::SeqCst);
            while current > observed_max
                && max_in_flight
                    .compare_exchange(observed_max, current, Ordering::SeqCst, Ordering::SeqCst)
                    .is_err()
            {
                observed_max = max_in_flight.load(Ordering::SeqCst);
            }

            thread::sleep(Duration::from_millis(5));
            in_flight.fetch_sub(1, Ordering::SeqCst);
            genes.iter().map(GeneValue::to_f64).sum::<f64>()
        }
    };

    let mut ga = EngineKernel::new(config, fitness).unwrap();
    ga.initialize_population().unwrap();

    let _ = ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| ga.evaluate_population());

    assert!(
        max_in_flight.load(Ordering::SeqCst) > 1,
        "fitness evaluation did not overlap across threads"
    );
}
