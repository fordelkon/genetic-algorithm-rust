use genetic_algorithm_rust::ga::core::{individual::Individual, population::Population};
use genetic_algorithm_rust::ga::operators::selection;
use genetic_algorithm_rust::{GeneValue, SelectionType};
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
            .all(|parent| parent.genes.len() == 2 && parent.fitness.is_some())
    );
}

#[test]
fn steady_state_selection_returns_top_individuals() {
    let mut rng = StdRng::seed_from_u64(9);
    let parents =
        selection::select_parents(&population(), &SelectionType::SteadyState, 2, &mut rng);

    assert_eq!(parents.len(), 2);
    assert_eq!(parents[0].fitness, Some(4.0));
    assert_eq!(parents[1].fitness, Some(3.0));
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
            .all(|parent| parent.genes.len() == 2 && parent.fitness.is_some())
    );
}
