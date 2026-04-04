use crate::ga::{
    crossover::CrossoverType,
    error::GaError,
    gene::{GeneScalarType, GeneValue, GenesDomain, GenesValueType},
    mutation::MutationType,
    selection::SelectionType,
    stop::StopCondition,
};

/// 遗传算法引擎的核心配置参数。
///
/// `GaConfig` 描述一次 GA 运行所需的全部静态配置，主要分为：
/// - 种群规模与迭代次数等基础参数
/// - 基因初始化范围、基因类型、基因搜索空间
/// - 父代选择、交叉、变异等进化策略
/// - 提前停止条件与随机种子
///
/// 这个结构体本身不执行遗传算法逻辑，只负责提供一份“合法、完整、可复现”的运行配置。
/// 推荐通过 [`GaConfigBuilder`]（使用 [`GaConfig::builder`] 方法）来构建此配置。
#[derive(Debug, Clone)]
pub struct GaConfig {
    // ==========================================
    // 基础参数 (Basic Parameters)
    // ==========================================
    /// 种群规模，即每一代保留的个体总数。
    ///
    /// 一般来说：
    /// - 值过小，搜索空间覆盖不足，容易早熟收敛。
    /// - 值过大，适应度（fitness）评估成本会明显上升。
    ///
    /// **约束**：合法值必须大于 `0`。
    pub population_size: usize,

    /// 染色体长度，即每个个体包含的基因数量。
    ///
    /// 通常等于待优化变量的个数。
    /// **约束**：合法值必须大于 `0`。
    pub num_genes: usize,

    /// 算法允许执行的最大迭代代数。
    ///
    /// 当 `stop_condition` 未提前触发时，算法在达到该代数后停止。
    /// **约束**：合法值必须大于 `0`。
    pub num_generations: usize,

    // ==========================================
    // 初始化与空间定义 (Initialization & Space)
    // ==========================================
    /// 未指定 `genes_domain` 时，随机初始化基因的默认下界。
    ///
    /// 该值也可作为某些变异（mutation）策略的默认数值参考范围。
    pub init_range_low: f64,

    /// 未指定 `genes_domain` 时，随机初始化基因的默认上界。
    pub init_range_high: f64,

    /// 染色体所有基因值类型定义。
    ///
    /// 它可以表示：
    /// - 所有基因统一为同一种类型。
    /// - 每个基因分别定义自己的类型。
    pub genes_value_type: GenesValueType,

    /// 染色体所有基因的取值空间。
    ///
    /// 它可以表示：
    /// - 所有基因共享一个全局搜索空间。
    /// - 每个基因有独立的搜索空间。
    ///
    /// 若为 `None`，则退回到 `init_range_low..=init_range_high`。
    pub genes_domain: Option<GenesDomain>,

    // ==========================================
    // 选择算子 (Selection)
    // ==========================================
    /// 每一代参与交配的父代个体数量。
    ///
    /// **约束**：
    /// - 必须大于等于 `2`。
    /// - 必须小于等于 `population_size`。
    pub num_parents_mating: usize,

    /// 父代选择策略。
    ///
    /// 例如轮盘赌选择、锦标赛选择、排序选择等。
    pub selection_type: SelectionType,

    /// 精英保留数量。
    ///
    /// 每一代中适应度最高的 `elitism_count` 个个体会直接进入下一代，
    /// 不经过交叉与变异。这可以避免当前最优解在进化过程中被破坏。
    ///
    /// **约束**：合法值必须不大于 `population_size`。
    pub elitism_count: usize,

    // ==========================================
    // 交叉算子 (Crossover)
    // ==========================================
    /// 交叉策略。
    ///
    /// 例如单点交叉、两点交叉、均匀交叉等。
    pub crossover_type: CrossoverType,

    /// 交叉发生的概率，范围为 `[0.0, 1.0]`。
    ///
    /// 常见取值在 `0.7..=0.95`。
    /// 如果为 `0.0`，则不会触发交叉，子代通常会直接复制父代基因。
    pub crossover_probability: f64,

    // ==========================================
    // 变异算子 (Mutation)
    // ==========================================
    /// 变异策略。
    ///
    /// 例如随机重置、随机扰动等。
    pub mutation_type: MutationType,

    /// 基因发生变异的概率，范围为 `[0.0, 1.0]`。
    ///
    /// 当 [`Self::mutation_num_genes`] 为 `None` 时，
    /// 该概率用于数值型变异算子的“逐基因”触发判断。
    /// 对于重排型变异算子（如交换、乱序、反转），该概率通常按“逐个体”判断。
    /// 通常不建议设得过高，否则算法会逐渐退化为纯随机搜索。
    pub mutation_probability: f64,

    /// 每个个体固定执行变异的基因数量。
    ///
    /// 当该值为 `Some(k)` 时，数值型变异算子会在每个个体上随机选择 `k` 个不同基因执行变异，
    /// 并覆盖默认的按 [`Self::mutation_probability`] 逐基因判断逻辑。
    /// 如果当前变异策略本身是自适应数值型变异，
    /// 则每个个体的变异基因数由算子内部参数动态决定，此字段不会生效。
    /// 当该值为 `None` 时，保持按概率逐基因触发的默认行为。
    ///
    /// **约束**：若设置为 `Some(k)`，则必须满足 `k <= num_genes`。
    pub mutation_num_genes: Option<usize>,

    // ==========================================
    // 控制与复现 (Control & Reproducibility)
    // ==========================================
    /// 随机数种子。
    ///
    /// 如果为 `Some(seed)`，则初始化、选择、交叉、变异都可复现，
    /// 非常适合调试和实验复现。
    pub random_seed: Option<u64>,

    /// 运行停止条件。
    ///
    /// 例如：到达最大代数、达到目标适应度、连续若干代最优值没有提升等。
    pub stop_condition: StopCondition,
}

/// [`GaConfig`] 的建造者，用于链式初始化配置。
///
/// 推荐优先使用 builder，而不是直接手写结构体字面量。
/// 这样可以：
/// - 减少初始化样板代码（内置了合理的默认值）。
/// - 只覆写真正关心的字段。
/// - 在 `build()` 时统一做完整的参数合法性校验。
///
/// # 示例 (Examples)
///
/// ```rust
/// # use genetic_algorithm_rust::ga::{config::GaConfig, crossover::CrossoverType, mutation::MutationType};
/// let config = GaConfig::builder(100, 10, 50, 50)
///     .crossover(CrossoverType::TwoPoint, 0.85)
///     .mutation_probability(0.1)
///     .init_range(-10.0, 10.0)
///     .build()
///     .expect("配置参数不合法");
/// ```
#[derive(Debug, Clone)]
pub struct GaConfigBuilder {
    config: GaConfig,
}

impl GaConfig {
    /// 创建一个 [`GaConfigBuilder`]。
    ///
    /// 必须提供一次 GA 运行最核心的四个参数：
    /// * `population_size` - 种群规模
    /// * `num_genes` - 基因数量（染色体长度）
    /// * `num_generations` - 最大迭代代数
    /// * `num_parents_mating` - 参与交配的父代数量
    pub fn builder(
        population_size: usize,
        num_genes: usize,
        num_generations: usize,
        num_parents_mating: usize,
    ) -> GaConfigBuilder {
        GaConfigBuilder::new(
            population_size,
            num_genes,
            num_generations,
            num_parents_mating,
        )
    }

    /// 校验当前配置是否合法。
    ///
    /// # 错误 (Errors)
    ///
    /// 如果配置不满足算法的逻辑约束，将返回 [`GaError::InvalidConfig`] 或 [`GaError::UnsupportedGeneType`]。
    /// 检查的内容包括但不限于：
    /// - 种群规模、基因数、代数是否大于 `0`。
    /// - 父代数量、精英数量是否越界（不能超过种群规模等）。
    /// - 概率参数是否处于 `[0.0, 1.0]` 区间。
    /// - `mutation_num_genes` 是否超过 `num_genes`。
    /// - 自适应变异中声明的低/高质量个体变异基因数是否超过 `num_genes`。
    /// - `genes_value_type` 与 `genes_domain` 的定义长度是否与 `num_genes` 匹配。
    /// - 无符号基因类型在没有 domain 的情况下，是否被赋予了负数的初始化下界。
    pub fn validate(&self) -> Result<(), GaError> {
        if self.population_size == 0 {
            return Err(GaError::InvalidConfig(
                "population_size must be greater than 0".into(),
            ));
        }

        if self.num_genes == 0 {
            return Err(GaError::InvalidConfig(
                "num_genes must be greater than 0".into(),
            ));
        }

        if self.num_generations == 0 {
            return Err(GaError::InvalidConfig(
                "num_generations must be greater than 0".into(),
            ));
        }

        if self.num_parents_mating < 2 || self.num_parents_mating > self.population_size {
            return Err(GaError::InvalidConfig(
                "num_parents_mating must be between 2 and population_size".into(),
            ));
        }

        if self.elitism_count > self.population_size {
            return Err(GaError::InvalidConfig(
                "elitism_count must not exceed population_size".into(),
            ));
        }

        if !(0.0..=1.0).contains(&self.crossover_probability) {
            return Err(GaError::InvalidConfig(
                "crossover_probability must be between 0 and 1".into(),
            ));
        }

        if !(0.0..=1.0).contains(&self.mutation_probability) {
            return Err(GaError::InvalidConfig(
                "mutation_probability must be between 0 and 1".into(),
            ));
        }

        if let Some(mutation_num_genes) = self.mutation_num_genes
            && mutation_num_genes > self.num_genes
        {
            return Err(GaError::InvalidConfig(
                "mutation_num_genes must not exceed num_genes".into(),
            ));
        }

        match &self.genes_value_type {
            GenesValueType::PerGene(types) if types.len() != self.num_genes => {
                return Err(GaError::InvalidConfig(
                    "per-gene genes_value_type length must match num_genes".into(),
                ));
            }
            GenesValueType::All(scalar_type) if !scalar_type.is_supported() => {
                return Err(GaError::UnsupportedGeneType(scalar_type.as_str().into()));
            }
            GenesValueType::PerGene(types) => {
                for scalar_type in types {
                    if !scalar_type.is_supported() {
                        return Err(GaError::UnsupportedGeneType(scalar_type.as_str().into()));
                    }
                }
            }
            _ => {}
        }

        match &self.genes_domain {
            Some(GenesDomain::PerGene(domains)) if domains.len() != self.num_genes => {
                return Err(GaError::InvalidConfig(
                    "per-gene genes_domain length must match num_genes".into(),
                ));
            }
            Some(space) => {
                for gene_index in 0..self.num_genes {
                    let domain = space.domain_for(gene_index);
                    if let Err(message) = domain.validate() {
                        return Err(GaError::InvalidConfig(message));
                    }

                    domain.validate_for_type(self.gene_value_type_for(gene_index))?;
                }
            }
            None => {}
        }

        for gene_index in 0..self.num_genes {
            let scalar_type = self.gene_value_type_for(gene_index);
            if scalar_type.is_unsigned() && self.init_range_low < 0.0 && self.genes_domain.is_none()
            {
                return Err(GaError::InvalidConfig(format!(
                    "init_range_low {} cannot be represented by {} without genes_domain",
                    self.init_range_low,
                    scalar_type.as_str()
                )));
            }
        }

        match &self.selection_type {
            SelectionType::Tournament { k } if *k == 0 => {
                return Err(GaError::InvalidConfig(
                    "tournament k must be greater than 0".into(),
                ));
            }
            _ => {}
        }

        match &self.mutation_type {
            MutationType::AdaptiveRandomReset {
                low_quality_num_genes,
                high_quality_num_genes,
                ..
            }
            | MutationType::AdaptiveRandomPerturbation {
                low_quality_num_genes,
                high_quality_num_genes,
                ..
            } => {
                if *low_quality_num_genes > self.num_genes
                    || *high_quality_num_genes > self.num_genes
                {
                    return Err(GaError::InvalidConfig(
                        "adaptive mutation gene counts must not exceed num_genes".into(),
                    ));
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// 返回指定基因索引对应的值类型。
    ///
    /// 当 `genes_value_type` 为统一类型时，所有索引都会返回同一个类型；
    /// 当 `genes_value_type` 为逐基因配置时，返回该位置的具体类型。
    pub fn gene_value_type_for(&self, gene_index: usize) -> GeneScalarType {
        self.genes_value_type.value_type_for(gene_index)
    }

    /// 为指定基因索引采样一个合法初始值。
    ///
    /// 采样顺序为：
    /// 1. 若存在 `genes_domain`，优先从该空间采样。
    /// 2. 否则从 `init_range_low..=init_range_high` 采样。
    /// 3. 最后再根据 `genes_value_type` 和 `genes_domain` 做一次归一化。
    ///
    /// # 恐慌 (Panics)
    ///
    /// 如果采样并尝试归一化的最终值无法被强制转换为合法的基因类型，函数将会 `panic`。
    /// 通常在 `GaConfig::validate()` 被成功调用后，此情况不应发生。
    pub fn sample_gene(&self, gene_index: usize, rng: &mut impl rand::Rng) -> GeneValue {
        let sampled = match &self.genes_domain {
            Some(space) => space.domain_for(gene_index).sample_numeric(rng),
            None => rng.gen_range(self.init_range_low..=self.init_range_high),
        };

        self.normalize_gene(gene_index, sampled)
            .expect("gene sampling should always produce a valid value")
    }

    /// 将一个浮点数原始值归一化为当前配置下的合法基因值。
    ///
    /// 归一化顺序为：
    /// 1. 若存在 `genes_domain`，先将结果映射/截断/吸附到合法搜索空间内。
    /// 2. 再按 `genes_value_type` 做类型修正与转换，例如转换为具体的整数或浮点数类型。
    ///
    /// # 错误 (Errors)
    ///
    /// 若输入的值超出了目标类型的表示范围（例如将负数转换为无符号类型，或将过大的数转换为小范围整数），
    /// 将返回 [`GaError`] 错误。
    pub fn normalize_gene(&self, gene_index: usize, value: f64) -> Result<GeneValue, GaError> {
        let scalar_type = self.gene_value_type_for(gene_index);
        let typed = match &self.genes_domain {
            Some(space) => space.domain_for(gene_index).normalize_numeric(value),
            None => value,
        };

        GeneValue::cast_from_f64(scalar_type, typed)
    }
}

impl GaConfigBuilder {
    /// 实例化一个携带完善“默认兜底策略”的建造器。
    ///
    /// 内置的默认行为策略为：
    /// - **基因范围**：`-4.0..=4.0`
    /// - **基因类型**：全部固定为双精度浮点 `F64`
    /// - **选择算子**：锦标赛选择 `Tournament { k: 3 }`
    /// - **交叉算子**：单点交叉 `SinglePoint`，概率 `0.8`
    /// - **变异算子**：随机扰动 `RandomPerturbation(-1.0, 1.0)`，概率 `0.05`
    /// - **固定变异基因数**：默认关闭（`None`）
    /// - **精英保留**：数量保留 `1`
    /// - **停止条件**：仅判断最大迭代代数 `MaxGenerations`
    ///
    /// # 参数 (Arguments)
    ///
    /// * `population_size` - 种群规模。
    /// * `num_genes` - 染色体基因数量。
    /// * `num_generations` - 最大迭代代数。
    /// * `num_parents_mating` - 繁殖期交配的父代数量。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回具备基础可运行状态的 [`GaConfigBuilder`] 实例。
    pub fn new(
        population_size: usize,
        num_genes: usize,
        num_generations: usize,
        num_parents_mating: usize,
    ) -> Self {
        Self {
            config: GaConfig {
                population_size,
                num_genes,
                num_generations,
                num_parents_mating,
                init_range_low: -4.0,
                init_range_high: 4.0,
                genes_value_type: GenesValueType::All(GeneScalarType::F64),
                genes_domain: None,
                selection_type: SelectionType::Tournament { k: 3 },
                elitism_count: 1,
                crossover_type: CrossoverType::SinglePoint,
                crossover_probability: 0.8,
                mutation_type: MutationType::RandomPerturbation {
                    min_delta: -1.0,
                    max_delta: 1.0,
                },
                mutation_probability: 0.05,
                mutation_num_genes: None,
                random_seed: None,
                stop_condition: StopCondition::MaxGenerations,
            },
        }
    }

    /// 设置未指定 `genes_domain` 时的默认随机初始化基因上下界。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `low` - 初始化的默认下界。
    /// * `high` - 初始化的默认上界。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn init_range(mut self, low: f64, high: f64) -> Self {
        self.config.init_range_low = low;
        self.config.init_range_high = high;
        self
    }

    /// 设置基因的数据类型定义（支持全局统一类型或逐个基因独立配置）。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `genes_value_type` - 描述基因标量类型的枚举配置。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn genes_value_type(mut self, genes_value_type: GenesValueType) -> Self {
        self.config.genes_value_type = genes_value_type;
        self
    }

    /// 设置基因的具体搜索空间及跨度域（可选）。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `genes_domain` - 定义搜索空间的枚举，若为 `None` 则退回使用 `init_range`。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn genes_domain(mut self, genes_domain: Option<GenesDomain>) -> Self {
        self.config.genes_domain = genes_domain;
        self
    }

    /// 设置优胜劣汰的父代选择策略。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `selection_type` - 选用的选择算法策略枚举。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn selection_type(mut self, selection_type: SelectionType) -> Self {
        self.config.selection_type = selection_type;
        self
    }

    /// 设置直通下一代的精英保留个体数量。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `elitism_count` - 不经历交叉变异，直接保留到下一代的顶级个体数。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn elitism_count(mut self, elitism_count: usize) -> Self {
        self.config.elitism_count = elitism_count;
        self
    }

    /// 单独设置基因交叉策略方案。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `crossover_type` - 选用的交叉操作算法枚举。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn crossover_type(mut self, crossover_type: CrossoverType) -> Self {
        self.config.crossover_type = crossover_type;
        self
    }

    /// 单独设置基因发生交叉的概率。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `crossover_probability` - 触发交叉的概率，有效区间 `[0.0, 1.0]`。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn crossover_probability(mut self, crossover_probability: f64) -> Self {
        self.config.crossover_probability = crossover_probability;
        self
    }

    /// 快捷方法：一次性同时设置交叉策略与交叉概率。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `crossover_type` - 选用的交叉操作算法枚举。
    /// * `crossover_probability` - 触发交叉的浮点数概率 `[0.0, 1.0]`。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn crossover(mut self, crossover_type: CrossoverType, crossover_probability: f64) -> Self {
        self.config.crossover_type = crossover_type;
        self.config.crossover_probability = crossover_probability;
        self
    }

    /// 单独设置基因变异策略方案。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `mutation_type` - 选用的变异操作算法枚举。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn mutation_type(mut self, mutation_type: MutationType) -> Self {
        self.config.mutation_type = mutation_type;
        self
    }

    /// 单独设置变异触发概率。
    ///
    /// 当 [`GaConfig::mutation_num_genes`] 为 `None` 时，
    /// 该值控制数值型变异算子的逐基因触发概率。
    /// 对于重排型变异算子（如 `Swap`/`Scramble`/`Inversion`），
    /// 该值表示每个个体执行一次该算子的概率。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `mutation_probability` - 变异触发概率，有效区间 `[0.0, 1.0]`。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn mutation_probability(mut self, mutation_probability: f64) -> Self {
        self.config.mutation_probability = mutation_probability;
        self
    }

    /// 设置每个个体固定要变异的基因数量。
    ///
    /// 当设置该值后，数值型变异算子会对每个个体随机挑选固定数量的不同基因执行变异。
    /// 若希望恢复默认的按概率逐基因触发模式，请不要调用此方法。
    /// 该配置当前只影响数值型变异算子，不影响 `Swap`、`Scramble`、`Inversion` 这类重排型算子。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `mutation_num_genes` - 每个个体固定变异的基因数，必须不大于 `num_genes`。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn mutation_num_genes(mut self, mutation_num_genes: usize) -> Self {
        self.config.mutation_num_genes = Some(mutation_num_genes);
        self
    }

    /// 快捷方法：一次性同时设置变异策略与变异触发概率。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `mutation_type` - 选用的变异操作算法枚举。
    /// * `mutation_probability` - 变异触发概率 `[0.0, 1.0]`。
    ///   它与 [`Self::mutation_num_genes`] 的关系与 [`Self::mutation_probability`] 方法一致。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn mutation(mut self, mutation_type: MutationType, mutation_probability: f64) -> Self {
        self.config.mutation_type = mutation_type;
        self.config.mutation_probability = mutation_probability;
        self
    }

    /// 设置随机生成器种子，以保证实验过程的精确复现。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `random_seed` - 固定的 `u64` 随机种子，若为 `None` 则使用系统级随机源。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn random_seed(mut self, random_seed: Option<u64>) -> Self {
        self.config.random_seed = random_seed;
        self
    }

    /// 设置算法的高级终止控制条件。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `stop_condition` - 决定算法何时停止演化的策略。
    ///
    /// # 返回值 (Returns)
    ///
    /// 返回构建器自身 `Self` 以支持链式调用。
    pub fn stop_condition(mut self, stop_condition: StopCondition) -> Self {
        self.config.stop_condition = stop_condition;
        self
    }

    /// 结束链式调用，构造最终的 [`GaConfig`] 并执行严格的参数验证。
    ///
    /// # 返回值 (Returns)
    ///
    /// 如果所有参数设置都合法且无逻辑冲突，返回验证通过的 [`GaConfig`] 实例。
    ///
    /// # 错误 (Errors)
    ///
    /// 若在构建链中设置的参数（或所沿用的默认参数）互相冲突、不符合数学或算法逻辑约束，
    /// 此方法内部将调用 `GaConfig::validate` 并抛出对应的 [`GaError`]。
    pub fn build(self) -> Result<GaConfig, GaError> {
        self.config.validate()?;
        Ok(self.config)
    }
}
