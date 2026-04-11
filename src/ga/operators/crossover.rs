use rand::Rng;

use crate::ga::core::{gene::GeneValue, individual::Individual};

/// 定义了遗传算法中可用的交叉（重组）策略。
///
/// 交叉算子用于将两个父代个体的基因序列组合在一起，从而生成新的子代个体。
/// 不同的交叉策略适用于不同的问题空间：对于强相关性基因，单点/两点交叉能较好地保留基因块；
/// 对于相互独立的基因，均匀交叉能提供更好的搜索空间探索能力。
#[derive(Debug, Clone)]
pub enum CrossoverType {
    /// 无交叉。子代将直接复制父代 1 的基因，不发生任何混合。
    None,

    /// 单点交叉（Single-point Crossover）。
    ///
    /// 在染色体上随机选择一个交叉点。子代在该点之前的基因来自父代 1，该点及之后的基因来自父代 2。
    SinglePoint,

    /// 两点交叉（Two-point Crossover）。
    ///
    /// 在染色体上随机选择两个交叉点。子代的基因序列在两点之间来自父代 2，其余两端来自父代 1。
    TwoPoint,

    /// 均匀交叉（Uniform Crossover）。
    ///
    /// 子代的每一个基因都有 50% 的概率来自父代 1，50% 的概率来自父代 2。
    Uniform,
}

/// 执行交叉操作，从父代群体中繁育出指定数量的新子代。
///
/// 该函数采用环形配对（Ring Pairing）策略：即第 `i` 个子代的父母分别为
/// `parents[i % len]` 和 `parents[(i + 1) % len]`。
///
/// 每次繁育是否触发交叉，由给定的 `crossover_probability` 决定。
/// 如果未触发交叉（或策略为 `CrossoverType::None`），子代将完全继承父代 1 的基因。
///
/// # 参数 (Arguments)
///
/// * `parents` - 参与交配的父代个体集合切片。
/// * `crossover_type` - 指定使用的交叉策略（见 [`CrossoverType`]）。
/// * `crossover_probability` - 触发交叉操作的概率，范围应在 `[0.0, 1.0]` 之间。
/// * `offspring_count` - 期望生成的子代个体总数。
/// * `rng` - 随机数生成器。
///
/// # 恐慌 (Panics)
///
/// 如果传入的 `parents` 切片为空（长度为 0），此函数内部的取模运算（`idx % parents.len()`）
/// 将触发**除以零（Division by zero）**的 Panic。
pub fn crossover(
    parents: &[Individual],
    crossover_type: &CrossoverType,
    crossover_probability: f64,
    offspring_count: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    let mut offspring = Vec::with_capacity(offspring_count);

    for idx in 0..offspring_count {
        // 环形配对选取父母
        let parent1 = &parents[idx % parents.len()];
        let parent2 = &parents[(idx + 1) % parents.len()];

        // 按概率决定是否进行基因重组
        let genes = if rng.gen_bool(crossover_probability) {
            match crossover_type {
                CrossoverType::None => parent1.genes.clone(),
                CrossoverType::SinglePoint => single_point(&parent1.genes, &parent2.genes, rng),
                CrossoverType::TwoPoint => two_point(&parent1.genes, &parent2.genes, rng),
                CrossoverType::Uniform => uniform(&parent1.genes, &parent2.genes, rng),
            }
        } else {
            // 未触发交叉，直接克隆 parent1 的基因
            parent1.genes.clone()
        };

        // 构造新个体（新的子代默认没有适应度，fitness 为 None）
        offspring.push(Individual::new(genes));
    }

    offspring
}

/// （内部实现）单点交叉。
///
/// 随机选取切割点 `split`，拼接 `parent1[..split]` 和 `parent2[split..]`。
/// 如果基因长度 `<=` 1，则无法切割，直接退化为复制 `parent1`。
fn single_point(
    parent1: &[GeneValue],
    parent2: &[GeneValue],
    rng: &mut impl Rng,
) -> Vec<GeneValue> {
    if parent1.len() <= 1 {
        return parent1.to_vec();
    }

    let split = rng.gen_range(1..parent1.len());
    parent1[..split]
        .iter()
        .chain(parent2[split..].iter())
        .cloned()
        .collect()
}

/// （内部实现）两点交叉。
///
/// 随机选取两个不同的切割点 `point1` 和 `point2`。
/// 拼接模式为：`parent1` -> `parent2` -> `parent1`。
/// 如果基因长度 `<=` 2，无法切出两个点，将自动降级并退化为单点交叉。
fn two_point(parent1: &[GeneValue], parent2: &[GeneValue], rng: &mut impl Rng) -> Vec<GeneValue> {
    if parent1.len() <= 2 {
        return single_point(parent1, parent2, rng);
    }

    let point1 = rng.gen_range(1..parent1.len() - 1);
    let point2 = rng.gen_range(point1 + 1..parent1.len());
    let mut child = Vec::with_capacity(parent1.len());

    child.extend_from_slice(&parent1[..point1]);
    child.extend_from_slice(&parent2[point1..point2]);
    child.extend_from_slice(&parent1[point2..]);
    child
}

/// （内部实现）均匀交叉。
///
/// 逐个遍历基因位，每一位掷硬币（50%概率），决定取自 `parent1` 还是 `parent2`。
fn uniform(parent1: &[GeneValue], parent2: &[GeneValue], rng: &mut impl Rng) -> Vec<GeneValue> {
    parent1
        .iter()
        .zip(parent2.iter())
        .map(|(left, right)| {
            if rng.gen_bool(0.5) {
                left.clone()
            } else {
                right.clone()
            }
        })
        .collect()
}
