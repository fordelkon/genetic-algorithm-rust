use rand::{SeedableRng, rngs::StdRng};

use genetic_algorithm_rust::ga::operators::crossover::{self, CrossoverType};
use genetic_algorithm_rust::ga::core::{gene::GeneValue, individual::Individual};

#[test]
fn crossover_none_keeps_parent_genes() {
    let parents = vec![
        Individual::new(vec![
            GeneValue::F64(1.0),
            GeneValue::F64(2.0),
            GeneValue::F64(3.0),
        ]),
        Individual::new(vec![
            GeneValue::F64(4.0),
            GeneValue::F64(5.0),
            GeneValue::F64(6.0),
        ]),
    ];
    let mut rng = StdRng::seed_from_u64(3);

    let offspring = crossover::crossover(&parents, &CrossoverType::None, 1.0, 2, &mut rng);

    assert_eq!(offspring[0].genes, parents[0].genes);
    assert_eq!(offspring[1].genes, parents[1].genes);
}

#[test]
fn single_point_crossover_preserves_length() {
    let parents = vec![
        Individual::new(vec![
            GeneValue::F64(1.0),
            GeneValue::F64(2.0),
            GeneValue::F64(3.0),
            GeneValue::F64(4.0),
        ]),
        Individual::new(vec![
            GeneValue::F64(5.0),
            GeneValue::F64(6.0),
            GeneValue::F64(7.0),
            GeneValue::F64(8.0),
        ]),
    ];
    let mut rng = StdRng::seed_from_u64(4);

    let offspring = crossover::crossover(&parents, &CrossoverType::SinglePoint, 1.0, 4, &mut rng);

    assert!(offspring.iter().all(|child| child.genes.len() == 4));
}
