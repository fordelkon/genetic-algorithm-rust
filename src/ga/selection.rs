use rand::Rng;

use crate::ga::{individual::Individual, population::Population};

/// 定义了遗传算法中可用的父代选择策略。
///
/// 选择算子用于从当前种群中挑选出若干个体作为交配的父母。
/// 选择的过程通常是**有放回的（With Replacement）**，这意味着同一个优秀的个体
/// 可能会被多次选中并参与多次交配。
#[derive(Debug, Clone)]
pub enum SelectionType {
    /// 稳态选择（Steady-state Selection）。
    ///
    /// 直接按照适应度从高到低排序，选择前 `num_parents` 个个体作为父代。
    /// 这是一个确定性选择策略，选择压力较强，适合希望快速保留优秀个体的场景。
    SteadyState,

    /// 锦标赛选择（Tournament Selection）。
    ///
    /// 每次从种群中随机抽取 `k` 个个体，相互比拼适应度，选取其中适应度最高的一个作为父代。
    /// - `k`（锦标赛规模）越大，选择压力越大，越容易选出超级个体（但也更容易早熟收敛）。
    /// - 通常推荐 `k = 2` 或 `k = 3`。
    Tournament { k: usize },

    /// 轮盘赌选择（Roulette Wheel Selection / 适应度比例选择）。
    ///
    /// 个体被选中的概率与其适应度成正比。适应度越高的个体，在轮盘上占据的面积越大。
    ///
    /// **注意**：如果个体的适应度存在负数，算法会自动将其平移到正数区间。
    RouletteWheel,

    /// 排序选择（Rank Selection）。
    ///
    /// 将种群按适应度从高到低进行排序，根据排名（Rank）来分配选择概率，而不是直接使用真实的适应度值。
    /// 排名第一的个体获得最高权重，排名最后的个体获得最低权重（权重线性递减）。
    ///
    /// 这种方法可以有效缓解轮盘赌选择中“超级个体垄断”或“所有个体适应度相近导致盲目搜索”的问题。
    Rank,

    /// 随机通用采样（Stochastic Universal Sampling, SUS）。
    ///
    /// 与轮盘赌选择一样，个体被选中的概率与权重成正比；
    /// 区别在于该方法一次生成一组等距指针进行采样，因此抽样方差通常更小、结果更稳定。
    ///
    /// **注意**：如果个体的适应度存在负数，算法会自动将其平移到正数区间。
    StochasticUniversalSampling,
}

/// 执行选择操作，从当前种群中挑选出指定数量的父代个体。
///
/// 这是一个公共的调度函数，它会根据传入的 `selection_type` 分发到具体的选择算法实现。
/// 选出的父代个体将被完全克隆，原种群不受影响。
///
/// # 参数 (Arguments)
///
/// * `population` - 当前进行选择操作的种群。
/// * `selection_type` - 指定使用的选择策略（见 [`SelectionType`]）。
/// * `num_parents` - 需要挑选出的父代个体总数。
/// * `rng` - 随机数生成器。
///
/// # 恐慌 (Panics)
///
/// - 如果传入的种群为空（`population.len() == 0`），将会触发 panic。
/// - 如果种群中的个体存在未评估适应度的情况（即 `fitness` 为 `None`），
///   或者适应度为 `NaN` 导致无法比较，将会触发 panic。
pub fn select_parents(
    population: &Population,
    selection_type: &SelectionType,
    num_parents: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    match selection_type {
        SelectionType::SteadyState => steady_state_selection(population, num_parents),
        SelectionType::Tournament { k } => tournament_selection(population, num_parents, *k, rng),
        SelectionType::RouletteWheel => roulette_wheel_selection(population, num_parents, rng),
        SelectionType::Rank => rank_selection(population, num_parents, rng),
        SelectionType::StochasticUniversalSampling => {
            stochastic_universal_selection(population, num_parents, rng)
        }
    }
}

/// （内部实现）稳态选择。
///
/// 将种群按适应度从高到低排序后，直接返回前 `num_parents` 个个体。
fn steady_state_selection(population: &Population, num_parents: usize) -> Vec<Individual> {
    let mut ranked = population.individuals.clone();
    ranked.sort_by(|left, right| {
        right
            .fitness_or_panic()
            .partial_cmp(&left.fitness_or_panic())
            .expect("fitness comparison failed")
    });
    ranked.into_iter().take(num_parents).collect()
}

/// （内部实现）锦标赛选择。
fn tournament_selection(
    population: &Population,
    num_parents: usize,
    k: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    let mut parents = Vec::with_capacity(num_parents);
    let len = population.len();

    for _ in 0..num_parents {
        let mut best: Option<&Individual> = None;

        for _ in 0..k {
            let candidate = &population.individuals[rng.gen_range(0..len)];
            if best
                .as_ref()
                .is_none_or(|current| candidate.fitness_or_panic() > current.fitness_or_panic())
            {
                best = Some(candidate);
            }
        }

        parents.push(best.expect("tournament should select a parent").clone());
    }

    parents
}

/// （内部实现）轮盘赌选择。
///
/// **平移机制**：标准的轮盘赌要求所有权重必须大于 0。如果种群中最小适应度 `<= 0`，
/// 该算法会找出一个偏移量（`shift`），把所有适应度平移到正数范围内。
///
/// **降级回退**：如果计算出的总权重依然 `<= 0.0`（例如全是严重的负数或浮点精度问题），
/// 将会自动降级并回退到 `k=2` 的锦标赛选择，以保证算法不崩溃。
fn roulette_wheel_selection(
    population: &Population,
    num_parents: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    let Some((weights, total)) = positive_selection_weights(population) else {
        return tournament_selection(population, num_parents, 2, rng);
    };

    let mut parents = Vec::with_capacity(num_parents);
    for _ in 0..num_parents {
        let mut threshold = rng.gen_range(0.0..total);
        // 默认选中最后一个个体，防止浮点数精度误差导致未能命中任何目标
        let mut selected = &population.individuals[population.len() - 1];

        for (individual, weight) in population.individuals.iter().zip(weights.iter()) {
            threshold -= *weight;
            if threshold <= 0.0 {
                selected = individual;
                break;
            }
        }

        parents.push(selected.clone());
    }

    parents
}

/// （内部实现）随机通用采样（SUS）。
///
/// 该算法先基于适应度构建累计权重区间，然后以固定步长放置多个采样指针。
/// 相比普通轮盘赌，它可以在一次抽样中更均匀地覆盖概率空间。
///
/// **降级回退**：如果总权重 `<= 0.0`，将自动降级为 `k=2` 的锦标赛选择。
fn stochastic_universal_selection(
    population: &Population,
    num_parents: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    let Some((weights, total)) = positive_selection_weights(population) else {
        return tournament_selection(population, num_parents, 2, rng);
    };

    if num_parents == 0 {
        return Vec::new();
    }

    let step = total / num_parents as f64;
    let first_pointer = rng.gen_range(0.0..step);
    let mut parents = Vec::with_capacity(num_parents);
    let mut cumulative = 0.0;
    let mut selected_index = 0usize;

    for parent_idx in 0..num_parents {
        let pointer = first_pointer + step * parent_idx as f64;

        while selected_index < weights.len() - 1 && cumulative + weights[selected_index] < pointer {
            cumulative += weights[selected_index];
            selected_index += 1;
        }

        parents.push(population.individuals[selected_index].clone());
    }

    parents
}

/// （内部实现）排序选择。
///
/// 将种群完全克隆并按适应度降序排列。
/// 排名第一（索引 0）的个体权重最高，排名最后的个体权重为 1。
/// 然后基于这些线性递减的排名权重执行类似轮盘赌的抽取。
fn rank_selection(
    population: &Population,
    num_parents: usize,
    rng: &mut impl Rng,
) -> Vec<Individual> {
    let mut ranked = population.individuals.clone();
    ranked.sort_by(|left, right| {
        right
            .fitness_or_panic()
            .partial_cmp(&left.fitness_or_panic())
            .expect("fitness comparison failed")
    });

    // 生成排名权重，最好的个体获得长度为 len 的权重，最差的个体获得权重 1
    let weights = (1..=ranked.len())
        .rev()
        .map(|rank| rank as f64)
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    let mut parents = Vec::with_capacity(num_parents);

    for _ in 0..num_parents {
        let mut threshold = rng.gen_range(0.0..total);
        let mut selected = ranked
            .last()
            .expect("ranked population should not be empty");

        for (individual, weight) in ranked.iter().zip(weights.iter()) {
            threshold -= *weight;
            if threshold <= 0.0 {
                selected = individual;
                break;
            }
        }

        parents.push(selected.clone());
    }

    parents
}

/// 为基于权重的选择算法生成严格为正的权重。
///
/// 如果原始 fitness 中存在非正值，会统一进行平移；
/// 若平移后总权重仍非正，则返回 `None` 以便调用方执行回退策略。
fn positive_selection_weights(population: &Population) -> Option<(Vec<f64>, f64)> {
    let min_fitness = population
        .individuals
        .iter()
        .map(Individual::fitness_or_panic)
        .fold(f64::INFINITY, f64::min);

    let shift = if min_fitness <= 0.0 {
        -min_fitness + 1e-9
    } else {
        0.0
    };

    let weights = population
        .individuals
        .iter()
        .map(|individual| individual.fitness_or_panic() + shift)
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();

    if total <= 0.0 {
        None
    } else {
        Some((weights, total))
    }
}
