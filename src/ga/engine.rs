use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::ga::{
    config::GaConfig, crossover, error::GaError, gene::GeneValue, individual::Individual, mutation,
    population::Population, rng::build_rng, selection, stats::RunStats, stop,
};

/// 遗传算法的核心驱动引擎。
///
/// `GeneticAlgorithm` 负责统筹和调度一次完整进化过程的所有环节。
/// 它持有全局配置、随机数状态、当前种群状态以及历史统计数据。
///
/// # 泛型参数
///
/// * `F` - 适应度评估函数（Fitness Function / Objective Function）的类型。
///   它必须是一个闭包或函数指针，接收基因序列切片 `&[GeneValue]`，并返回一个 `f64` 类型的适应度评分。
///   **评分越高，代表该个体越优秀。**
pub struct GeneticAlgorithm<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync,
{
    /// 遗传算法的运行配置参数。
    pub config: GaConfig,
    /// 适应度评估函数（闭包）。用于计算每个个体的优劣程度。
    pub fitness_fn: F,
    /// 当前代的种群状态。
    pub population: Population,
    /// 当前已演化的代数（从 `0` 开始计数）。
    pub generation: usize,
    /// 算法运行过程中的统计信息（如历代最优适应度、平均适应度等）。
    pub stats: RunStats,
    /// 内部使用的标准随机数生成器。
    rng: StdRng,
}

impl<F> GeneticAlgorithm<F>
where
    F: Fn(&[GeneValue]) -> f64 + Sync,
{
    /// 创建并初始化一个新的遗传算法实例。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `config` - 遗传算法配置（推荐通过 `GaConfigBuilder` 构建）。
    /// * `fitness_fn` - 用户自定义的适应度评估函数。
    ///
    /// # 错误 (Errors)
    ///
    /// 在初始化之前，此方法会调用 `config.validate()`。如果配置参数不符合逻辑要求
    /// （例如概率越界、规模为0等），将返回 [`GaError::InvalidConfig`]。
    pub fn new(config: GaConfig, fitness_fn: F) -> Result<Self, GaError> {
        config.validate()?;
        let rng = build_rng(config.random_seed);

        Ok(Self {
            config,
            fitness_fn,
            population: Population::empty(),
            generation: 0,
            stats: RunStats::default(),
            rng,
        })
    }

    /// 随机初始化初代种群（第 0 代）。
    ///
    /// 根据 `config.population_size` 的大小，调用配置中的 `sample_gene` 方法，
    /// 为每一个个体在合法定义域内随机生成初始基因。
    ///
    /// 新生成的个体默认没有适应度（`fitness` 为 `None`）。
    pub fn initialize_population(&mut self) -> Result<(), GaError> {
        let mut individuals = Vec::with_capacity(self.config.population_size);

        for _ in 0..self.config.population_size {
            let genes = (0..self.config.num_genes)
                .map(|gene_index| self.config.sample_gene(gene_index, &mut self.rng))
                .collect::<Vec<_>>();
            individuals.push(Individual::new(genes));
        }

        self.population = Population::new(individuals);
        Ok(())
    }

    /// 对当前种群中的所有个体执行适应度评估。
    ///
    /// 该方法会并行遍历 `population` 中的每一个个体，将其基因序列提取并传递给
    /// 用户定义的 `fitness_fn`，计算结果会被写回个体的 `fitness` 字段中。
    pub fn evaluate_population(&mut self) {
        self.population
            .individuals
            .par_iter_mut()
            .for_each(|individual| {
                let fitness = (self.fitness_fn)(&individual.genes);
                individual.fitness = Some(fitness);
            });
    }

    /// 根据配置的选择策略，从当前种群中挑选父代。
    ///
    /// 返回的父代列表将被完全克隆，准备用于后续的交叉繁育。
    pub fn select_parents(&mut self) -> Vec<Individual> {
        selection::select_parents(
            &self.population,
            &self.config.selection_type,
            self.config.num_parents_mating,
            &mut self.rng,
        )
    }

    /// 对选出的父代执行交叉操作，生成子代。
    ///
    /// 生成的子代数量会自动扣除精英保留的数量
    /// （即 `offspring_count = population_size - elitism_count`），
    /// 以确保最终合并后的新种群规模恒定不变。
    pub fn crossover(&mut self, parents: &[Individual]) -> Vec<Individual> {
        let offspring_count = self
            .config
            .population_size
            .saturating_sub(self.config.elitism_count);
        crossover::crossover(
            parents,
            &self.config.crossover_type,
            self.config.crossover_probability,
            offspring_count,
            &mut self.rng,
        )
    }

    /// 对交叉生成的子代群体执行变异操作。
    ///
    /// 变异是原地（in-place）发生的。如果个体的基因发生了变异，其适应度将被重置为 `None`。
    ///
    /// 当变异策略为自适应变异时，子代在变异前会先被临时评估一次 fitness，
    /// 以便根据“低于平均适应度 / 高于或等于平均适应度”的规则决定本个体的变异强度。
    /// 这次临时评估仅用于本轮变异决策；变异完成后，个体会再次在正常评估阶段重新计算 fitness。
    pub fn mutate(&mut self, offspring: &mut [Individual]) {
        if matches!(
            self.config.mutation_type,
            mutation::MutationType::AdaptiveRandomReset { .. }
                | mutation::MutationType::AdaptiveRandomPerturbation { .. }
        ) {
            offspring.par_iter_mut().for_each(|individual| {
                let fitness = (self.fitness_fn)(&individual.genes);
                individual.fitness = Some(fitness);
            });
        }

        mutation::mutate(
            offspring,
            &self.config,
            &self.config.mutation_type,
            self.config.mutation_probability,
            &mut self.rng,
        );
    }

    /// 步进执行一个完整的世代演化（Generation）。
    ///
    /// 此方法封装了遗传算法单次迭代的核心流水线：
    /// 1. **选择（Selection）**：挑选优秀的父代。
    /// 2. **交叉（Crossover）**：繁育新的子代。
    /// 3. **变异（Mutation）**：对子代引入随机扰动。
    /// 4. **精英保留（Elitism）**：将上一代最优的个体直接保送到下一代。
    /// 5. **合并（Combine）**：组建出新一代种群。
    /// 6. **评估（Evaluation）**：重新计算全种群的适应度。
    /// 7. **统计（Record）**：记录本代的各项关键指标。
    ///
    /// # 错误 (Errors)
    ///
    /// 如果在记录统计信息时提取适应度失败，可能返回 [`GaError::EmptyPopulation`]。
    pub fn next_generation(&mut self) -> Result<(), GaError> {
        let parents = self.select_parents();
        let mut offspring = self.crossover(&parents);
        self.mutate(&mut offspring);

        // 提取精英个体（需要保证当前 population 已经计算过适应度）
        let mut next_individuals = self.population.elite(self.config.elitism_count);
        // 将产生的子代追加到精英列表后
        next_individuals.extend(offspring);

        // 防御性截断：确保无论计算是否有误差，种群规模严格符合配置
        if next_individuals.len() > self.config.population_size {
            next_individuals.truncate(self.config.population_size);
        }

        self.population = Population::new(next_individuals);
        self.evaluate_population();
        self.stats.record(&self.population)?;
        Ok(())
    }

    /// 启动并自动运行遗传算法，直到满足停止条件。
    ///
    /// 这是供外部调用的最高层接口。它会自动执行初始化，然后不断循环调用 `next_generation()`，
    /// 每迭代一次，都会将 `generation` 计数器加一。
    /// 当满足配置的 `stop_condition` 时，循环结束并返回最终的统计数据。
    ///
    /// # 返回 (Returns)
    ///
    /// 成功结束后，返回指向运行统计数据 [`RunStats`] 的引用。
    ///
    /// # 错误 (Errors)
    ///
    /// 运行过程中如果有任何内部步骤（如初始化或记录统计数据）失败，将立即中断并返回错误。
    pub fn run(&mut self) -> Result<&RunStats, GaError> {
        self.initialize_population()?;
        self.evaluate_population();
        self.stats.record(&self.population)?;

        // 检查停止条件
        while !stop::should_stop(
            &self.config.stop_condition,
            self.generation,
            self.best_solution()?.fitness_or_panic(),
            &self.stats,
            self.config.num_generations,
        ) {
            self.next_generation()?;
            self.generation += 1;
        }

        Ok(&self.stats)
    }

    /// 获取当前种群中适应度最高的个体（即当前找到的最优解）。
    ///
    /// # 错误 (Errors)
    ///
    /// 若当前种群尚未初始化（为空），将返回 [`GaError::EmptyPopulation`]。
    pub fn best_solution(&self) -> Result<&Individual, GaError> {
        self.population.best()
    }
}
