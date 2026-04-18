use genetic_algorithm_rust::ga::core::{individual::Individual, population::Population};
use genetic_algorithm_rust::ga::operators::{nsga2, selection};
use genetic_algorithm_rust::{Evaluation, GeneValue, OptimizationMode, SelectionType};
use rand::{SeedableRng, rngs::StdRng};

fn population() -> Population {
    Population::new(vec![
        Individual::with_fitness(vec![GeneValue::F64(1.0), GeneValue::F64(1.0)], 1.0),
        Individual::with_fitness(vec![GeneValue::F64(2.0), GeneValue::F64(2.0)], 2.0),
        Individual::with_fitness(vec![GeneValue::F64(3.0), GeneValue::F64(3.0)], 3.0),
        Individual::with_fitness(vec![GeneValue::F64(4.0), GeneValue::F64(4.0)], 4.0),
    ])
}

#[test]
fn tournament_selection_returns_requested_number() {
    let mut rng = StdRng::seed_from_u64(7);
    let parents = selection::select_parents(
        &population(),
        &SelectionType::Tournament { k: 2 },
        3,
        &mut rng,
    );

    assert_eq!(parents.len(), 3);
}

#[test]
fn rank_selection_returns_valid_parents() {
    let mut rng = StdRng::seed_from_u64(8);
    let parents = selection::select_parents(&population(), &SelectionType::Rank, 2, &mut rng);

    assert!(
        parents
            .iter()
            .all(|parent| parent.genes.len() == 2 && parent.evaluation.is_some())
    );
}

#[test]
fn steady_state_selection_returns_top_individuals() {
    let mut rng = StdRng::seed_from_u64(9);
    let parents =
        selection::select_parents(&population(), &SelectionType::SteadyState, 2, &mut rng);

    assert_eq!(parents.len(), 2);
    assert_eq!(parents[0].evaluation, Some(Evaluation::Single(4.0)));
    assert_eq!(parents[1].evaluation, Some(Evaluation::Single(3.0)));
}

#[test]
fn stochastic_universal_selection_returns_valid_parents() {
    let mut rng = StdRng::seed_from_u64(10);
    let parents = selection::select_parents(
        &population(),
        &SelectionType::StochasticUniversalSampling,
        3,
        &mut rng,
    );

    assert_eq!(parents.len(), 3);
    assert!(
        parents
            .iter()
            .all(|parent| parent.genes.len() == 2 && parent.evaluation.is_some())
    );
}

#[test]
fn select_survivors_uses_single_objective_priority() {
    let survivors =
        selection::select_survivors(&population(), &OptimizationMode::SingleObjective, 2).unwrap();

    assert_eq!(survivors.len(), 2);
    assert_eq!(survivors[0].evaluation, Some(Evaluation::Single(4.0)));
    assert_eq!(survivors[1].evaluation, Some(Evaluation::Single(3.0)));
}

#[test]
fn select_survivors_uses_nsga2_priority() {
    let mut population = Population::new(vec![
        Individual::with_objectives(vec![GeneValue::F64(0.0)], vec![0.1, 0.2]),
        Individual::with_objectives(vec![GeneValue::F64(1.0)], vec![0.2, 0.4]),
        Individual::with_objectives(vec![GeneValue::F64(2.0)], vec![0.4, 0.1]),
    ]);
    nsga2::assign_population_metadata(&mut population).unwrap();

    let survivors = selection::select_survivors(
        &population,
        &OptimizationMode::Nsga2 { num_objectives: 2 },
        2,
    )
    .unwrap();

    assert_eq!(survivors.len(), 2);
    assert!(
        survivors
            .iter()
            .all(|individual| individual.rank == Some(0))
    );
}
