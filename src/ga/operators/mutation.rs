use rand::{Rng, seq::SliceRandom};

use crate::ga::core::individual::Individual;
use crate::ga::engine::config::GaConfig;

/// 定义了遗传算法中可用的基因变异策略。
///
/// 变异操作用于在种群中引入新的基因特征，防止算法过早收敛到局部最优解。
#[derive(Debug, Clone)]
pub enum MutationType {
    /// 无变异。个体的基因组将保持原样，不发生任何改变。
    None,

    /// 随机重置变异（Random Resetting）。
    ///
    /// 当基因发生变异时，用一个全新的随机值替换原有的基因值。
    RandomReset {
        /// 随机生成新值的下界（包含）
        min: f64,
        /// 随机生成新值的上界（包含）
        max: f64,
    },

    /// 随机扰动变异（Random Perturbation）。
    ///
    /// 当基因发生变异时，在当前基因值的基础上，加上一个随机产生的偏移量（Delta）。
    RandomPerturbation {
        /// 随机偏移量的下界（包含），通常为负数
        min_delta: f64,
        /// 随机偏移量的上界（包含），通常为正数
        max_delta: f64,
    },

    /// 自适应随机重置变异（Adaptive Random Resetting）。
    ///
    /// 先根据个体适应度与当前待变异群体平均适应度的比较结果，
    /// 决定当前个体应当变异多少个基因：
    /// - 低于平均适应度：使用 `low_quality_num_genes`
    /// - 高于或等于平均适应度：使用 `high_quality_num_genes`
    ///
    /// 被选中的基因会被一个新的随机值替换。
    AdaptiveRandomReset {
        /// 随机生成新值的下界（包含）
        min: f64,
        /// 随机生成新值的上界（包含）
        max: f64,
        /// 低质量个体要变异的基因数
        low_quality_num_genes: usize,
        /// 高质量个体要变异的基因数
        high_quality_num_genes: usize,
    },

    /// 自适应随机扰动变异（Adaptive Random Perturbation）。
    ///
    /// 与 [`MutationType::AdaptiveRandomReset`] 相同，
    /// 变异基因数量会根据个体质量动态调整；
    /// 区别在于该算子会在基因原值基础上叠加随机偏移量。
    AdaptiveRandomPerturbation {
        /// 随机偏移量的下界（包含），通常为负数
        min_delta: f64,
        /// 随机偏移量的上界（包含），通常为正数
        max_delta: f64,
        /// 低质量个体要变异的基因数
        low_quality_num_genes: usize,
        /// 高质量个体要变异的基因数
        high_quality_num_genes: usize,
    },

    /// 交换变异（Swap Mutation）。
    ///
    /// 随机选择两个不同的基因位置并交换它们的值。
    Swap,

    /// 乱序变异（Scramble Mutation）。
    ///
    /// 随机选择一个连续区间，并将该区间内的基因顺序随机打乱。
    Scramble,

    /// 反转变异（Inversion Mutation）。
    ///
    /// 随机选择一个连续区间，并将该区间内的基因顺序反转。
    Inversion,
}

/// 对给定的个体切片（种群子集）执行变异操作。
///
/// 该函数会遍历每个个体的每一个基因，根据给定的 `mutation_probability`（变异概率）
/// 决定该基因是否发生变异。如果发生变异，则按照 `mutation_type` 指定的策略修改基因值。
/// 当 `config.mutation_num_genes` 为 `Some(k)` 且当前算子为数值型变异时，
/// 函数会改为在每个个体中固定随机选择 `k` 个不同基因执行变异。
/// 当当前算子为自适应数值型变异时，
/// 每个个体实际变异的基因数量会优先由其 fitness 相对群体平均值的位置决定，
/// 此时不会读取 `config.mutation_num_genes`。
/// 对于 [`MutationType::Swap`]、[`MutationType::Scramble`]、[`MutationType::Inversion`] 这类
/// 基于“重排”的变异算子，`mutation_probability` 表示“每个个体是否执行一次该算子”的概率，
/// 而不是逐基因判断。
///
/// **注意**：一旦个体的任何基因发生了变异，其原有的适应度（fitness）将会失效，
/// 函数会自动将其 `fitness` 重置为 `None`，以便在后续步骤中重新评估。
///
/// # 参数 (Arguments)
///
/// * `individuals` - 需要进行变异操作的个体列表的**可变借用**。
/// * `config` - 遗传算法的全局配置，提供基因的采样规则 (`sample_gene`) 和标准化边界限制 (`normalize_gene`)。
/// * `mutation_type` - 本次变异所采用的具体变异策略（见 [`MutationType`]）。
/// * `mutation_probability` - 变异触发概率，取值范围应在 `[0.0, 1.0]` 之间。
///   对数值型变异算子（如 [`MutationType::RandomReset`]、[`MutationType::RandomPerturbation`]），
///   当 `config.mutation_num_genes` 为 `None` 时，该概率按“每个基因”独立判断；
///   当 `config.mutation_num_genes` 为 `Some(_)` 时，该概率对这类算子不生效；
///   对自适应数值型变异算子（如 [`MutationType::AdaptiveRandomReset`]、
///   [`MutationType::AdaptiveRandomPerturbation`]），该概率同样不生效；
///   对重排型变异算子（如 [`MutationType::Swap`]、[`MutationType::Scramble`]、[`MutationType::Inversion`]），
///   该概率按“每个个体”判断一次。
/// * `rng` - 用于生成随机数和计算概率的随机数生成器。
///
/// # 恐慌 (Panics)
///
/// * 如果变异策略产生的新值超出了允许的有效范围，且 `config.normalize_gene` 返回 `Err` 时，此函数会触发 **Panic**。
///   （代码内部假定了 `normalize_gene` 总是能够将变异后的值成功纠正/映射为有效值）。
/// * 如果当前算子为自适应变异，但传入的个体没有预先评估 fitness，
///   在计算群体平均适应度或读取个体 fitness 时会触发 panic。
pub fn mutate(
    individuals: &mut [Individual],
    config: &GaConfig,
    mutation_type: &MutationType,
    mutation_probability: f64,
    rng: &mut impl Rng,
) {
    let average_fitness = adaptive_average_fitness(individuals, mutation_type);

    for individual in individuals {
        // 标记当前个体是否真正发生了基因改变
        let mut has_mutated = false;

        match adaptive_mutation_num_genes(individual, mutation_type, average_fitness)
            .or(config.mutation_num_genes)
        {
            Some(mutation_num_genes) => {
                has_mutated |= mutate_fixed_num_genes(
                    &mut individual.genes,
                    config,
                    mutation_type,
                    mutation_num_genes,
                    rng,
                );
            }
            None => {
                for (gene_index, gene) in individual.genes.iter_mut().enumerate() {
                    // 根据给定的概率决定当前基因是否变异
                    if !rng.gen_bool(mutation_probability) {
                        continue;
                    }

                    has_mutated |= mutate_gene(gene, gene_index, config, mutation_type, rng);
                }
            }
        }

        match mutation_type {
            MutationType::Swap if rng.gen_bool(mutation_probability) => {
                has_mutated |= swap_genes(&mut individual.genes, rng);
            }
            MutationType::Scramble if rng.gen_bool(mutation_probability) => {
                has_mutated |= scramble_genes(&mut individual.genes, rng);
            }
            MutationType::Inversion if rng.gen_bool(mutation_probability) => {
                has_mutated |= invert_genes(&mut individual.genes, rng);
            }
            _ => {}
        }

        // 如果个体发生了变异，清除其旧的适应度缓存
        if has_mutated {
            individual.fitness = None;
        }
    }
}

/// 计算自适应变异所需的群体平均适应度。
///
/// 只有当 `mutation_type` 为自适应变异算子时才会返回 `Some(avg)`；
/// 否则返回 `None`，表示当前调用不需要此信息。
///
/// # Panics
///
/// 如果 `mutation_type` 为自适应变异，但 `individuals` 中存在 `fitness == None` 的个体，
/// 则内部调用 [`Individual::fitness_or_panic`] 会触发 panic。
fn adaptive_average_fitness(
    individuals: &[Individual],
    mutation_type: &MutationType,
) -> Option<f64> {
    if !is_adaptive_mutation(mutation_type) || individuals.is_empty() {
        return None;
    }

    let total = individuals
        .iter()
        .map(Individual::fitness_or_panic)
        .sum::<f64>();
    Some(total / individuals.len() as f64)
}

/// 根据个体 fitness 相对平均值的位置，决定自适应变异应修改多少个基因。
///
/// 规则为：
/// - `fitness < average_fitness`：返回低质量个体对应的变异基因数
/// - `fitness >= average_fitness`：返回高质量个体对应的变异基因数
///
/// 非自适应算子会返回 `None`。
///
/// # Panics
///
/// 如果调用方为自适应变异传入了 `average_fitness`，但当前个体 `fitness == None`，
/// 则会触发 panic。
fn adaptive_mutation_num_genes(
    individual: &Individual,
    mutation_type: &MutationType,
    average_fitness: Option<f64>,
) -> Option<usize> {
    let average_fitness = average_fitness?;
    let fitness = individual.fitness_or_panic();

    match mutation_type {
        MutationType::AdaptiveRandomReset {
            low_quality_num_genes,
            high_quality_num_genes,
            ..
        }
        | MutationType::AdaptiveRandomPerturbation {
            low_quality_num_genes,
            high_quality_num_genes,
            ..
        } => Some(if fitness < average_fitness {
            *low_quality_num_genes
        } else {
            *high_quality_num_genes
        }),
        _ => None,
    }
}

/// 判断当前变异算子是否属于“需要预先读取 fitness”的自适应变异。
fn is_adaptive_mutation(mutation_type: &MutationType) -> bool {
    matches!(
        mutation_type,
        MutationType::AdaptiveRandomReset { .. } | MutationType::AdaptiveRandomPerturbation { .. }
    )
}

/// 对单个个体按“固定数量基因”模式执行数值型变异。
///
/// 该函数同时服务于两种场景：
/// - `config.mutation_num_genes` 驱动的固定基因数变异
/// - 自适应变异根据个体 fitness 动态得出的固定基因数变异
///
/// 如果当前 `mutation_type` 不属于数值型变异算子，则不会做任何修改。
///
/// # Returns
///
/// * `true` - 至少有一个基因被成功变异。
/// * `false` - 当前算子不适用、`mutation_num_genes == 0`，或没有可变异基因。
fn mutate_fixed_num_genes(
    genes: &mut [crate::ga::core::gene::GeneValue],
    config: &GaConfig,
    mutation_type: &MutationType,
    mutation_num_genes: usize,
    rng: &mut impl Rng,
) -> bool {
    match mutation_type {
        MutationType::RandomReset { .. }
        | MutationType::RandomPerturbation { .. }
        | MutationType::AdaptiveRandomReset { .. }
        | MutationType::AdaptiveRandomPerturbation { .. } => {}
        MutationType::None
        | MutationType::Swap
        | MutationType::Scramble
        | MutationType::Inversion => return false,
    }

    if mutation_num_genes == 0 || genes.is_empty() {
        return false;
    }

    let mut indices = (0..genes.len()).collect::<Vec<_>>();
    indices.shuffle(rng);

    let mut has_mutated = false;
    for gene_index in indices.into_iter().take(mutation_num_genes) {
        has_mutated |= mutate_gene(
            &mut genes[gene_index],
            gene_index,
            config,
            mutation_type,
            rng,
        );
    }

    has_mutated
}

/// 对单个基因执行一次数值型变异。
///
/// 该函数是 [`mutate`] 的底层原子操作，负责将某一个给定位置的基因
/// 按照 `mutation_type` 转换成新的合法值。
/// 它同时支持普通数值型变异和自适应数值型变异；
/// 二者的差异只体现在“选多少个基因进行变异”，而不是单个基因的变换方式。
///
/// # Returns
///
/// * `true` - 当前基因已被修改。
/// * `false` - 当前 `mutation_type` 不适用于单基因数值变异。
fn mutate_gene(
    gene: &mut crate::ga::core::gene::GeneValue,
    gene_index: usize,
    config: &GaConfig,
    mutation_type: &MutationType,
    rng: &mut impl Rng,
) -> bool {
    match mutation_type {
        MutationType::None => false,
        MutationType::RandomReset { min, max } => {
            *gene = if config.genes_domain.is_some() {
                config.sample_gene(gene_index, rng)
            } else {
                config
                    .normalize_gene(gene_index, rng.gen_range(*min..=*max))
                    .expect("random reset mutation should normalize into a valid value")
            };
            true
        }
        MutationType::AdaptiveRandomReset { min, max, .. } => {
            *gene = if config.genes_domain.is_some() {
                config.sample_gene(gene_index, rng)
            } else {
                config
                    .normalize_gene(gene_index, rng.gen_range(*min..=*max))
                    .expect("adaptive random reset mutation should normalize into a valid value")
            };
            true
        }
        MutationType::RandomPerturbation {
            min_delta,
            max_delta,
        } => {
            let mutated = gene.to_f64() + rng.gen_range(*min_delta..=*max_delta);
            *gene = config
                .normalize_gene(gene_index, mutated)
                .expect("random perturbation mutation should normalize into a valid value");
            true
        }
        MutationType::AdaptiveRandomPerturbation {
            min_delta,
            max_delta,
            ..
        } => {
            let mutated = gene.to_f64() + rng.gen_range(*min_delta..=*max_delta);
            *gene = config.normalize_gene(gene_index, mutated).expect(
                "adaptive random perturbation mutation should normalize into a valid value",
            );
            true
        }
        MutationType::Swap | MutationType::Scramble | MutationType::Inversion => false,
    }
}

/// 在一个染色体中随机交换两个不同位置的基因。
///
/// # Returns
///
/// * `true` - 成功执行了一次交换。
/// * `false` - 基因数量不足 2，无法完成交换。
fn swap_genes(genes: &mut [crate::ga::core::gene::GeneValue], rng: &mut impl Rng) -> bool {
    if genes.len() < 2 {
        return false;
    }

    let left = rng.gen_range(0..genes.len());
    let mut right = rng.gen_range(0..genes.len() - 1);
    if right >= left {
        right += 1;
    }

    genes.swap(left, right);
    true
}

/// 在一个随机选出的连续区间内原地打乱基因顺序。
///
/// 该操作只会重排基因位置，不会新增、删除或修改任何基因值本身。
///
/// # Returns
///
/// * `true` - 成功对某个区间执行了乱序。
/// * `false` - 基因数量不足 2，无法形成有效区间。
fn scramble_genes(genes: &mut [crate::ga::core::gene::GeneValue], rng: &mut impl Rng) -> bool {
    let Some((start, end)) = random_subsequence_bounds(genes.len(), rng) else {
        return false;
    };

    genes[start..=end].shuffle(rng);
    true
}

/// 在一个随机选出的连续区间内原地反转基因顺序。
///
/// 该操作只会改变区间内基因的排列方向，不会修改基因值本身。
///
/// # Returns
///
/// * `true` - 成功对某个区间执行了反转。
/// * `false` - 基因数量不足 2，无法形成有效区间。
fn invert_genes(genes: &mut [crate::ga::core::gene::GeneValue], rng: &mut impl Rng) -> bool {
    let Some((start, end)) = random_subsequence_bounds(genes.len(), rng) else {
        return false;
    };

    genes[start..=end].reverse();
    true
}

/// 随机生成一个合法的连续子区间边界。
///
/// 返回的 `(start, end)` 满足：
/// - `start < end`
/// - `end < len`
/// - 区间至少包含 2 个基因
///
/// # Returns
///
/// * `Some((start, end))` - 一个可用于重排/反转的闭区间边界。
/// * `None` - 当 `len < 2` 时，不存在有效区间。
fn random_subsequence_bounds(len: usize, rng: &mut impl Rng) -> Option<(usize, usize)> {
    if len < 2 {
        return None;
    }

    let start = rng.gen_range(0..len - 1);
    let end = rng.gen_range(start + 1..len);
    Some((start, end))
}
