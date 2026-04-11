use crate::ga::core::individual::Individual;
use crate::ga::error::GaError;

/// 遗传算法中的种群（Population）。
///
/// 种群表示在算法的某一次迭代（某一代）中，所有存活个体（[`Individual`]）的集合。
/// 提供了对个体集合的基础管理功能，以及基于适应度（fitness）的排序、筛选和统计操作。
#[derive(Debug, Clone)]
pub struct Population {
    /// 构成当前种群的个体列表。
    pub individuals: Vec<Individual>,
}

impl Population {
    /// 根据给定的个体集合创建一个新的种群实例。
    pub fn new(individuals: Vec<Individual>) -> Self {
        Self { individuals }
    }

    /// 创建一个没有任何个体的空种群。
    pub fn empty() -> Self {
        Self {
            individuals: Vec::new(),
        }
    }

    /// 返回当前种群中包含的个体总数。
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// 检查当前种群是否为空（即个体数量为 `0`）。
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// 查找并返回当前种群中适应度（fitness）最高的个体引用。
    ///
    /// # 错误 (Errors)
    ///
    /// 如果种群为空，将返回 [`GaError::EmptyPopulation`]。
    ///
    /// # 恐慌 (Panics)
    ///
    /// 此方法假定种群中的所有个体都已经完成了适应度计算，并且适应度是有效的数值。
    /// 在以下情况会触发 panic：
    /// 1. 种群中存在尚未计算适应度的个体（`fitness_or_panic` 触发）。
    /// 2. 个体的适应度为 `NaN`，导致浮点数比较失败（`expect` 触发）。
    pub fn best(&self) -> Result<&Individual, GaError> {
        self.individuals
            .iter()
            .max_by(|left, right| {
                left.fitness_or_panic()
                    .partial_cmp(&right.fitness_or_panic())
                    .expect("fitness comparison failed")
            })
            .ok_or(GaError::EmptyPopulation)
    }

    /// 对种群内的个体按适应度进行**降序**排序。
    ///
    /// 排序后，适应度最高的个体将位于向量的开头（索引为 `0` 的位置）。此操作会原地修改当前种群。
    ///
    /// # 恐慌 (Panics)
    ///
    /// 在以下情况会触发 panic：
    /// 1. 种群中存在尚未计算适应度的个体。
    /// 2. 存在适应度为 `NaN` 的个体，无法进行浮点数比较。
    pub fn sort_by_fitness_desc(&mut self) {
        self.individuals.sort_by(|left, right| {
            right
                .fitness_or_panic()
                .partial_cmp(&left.fitness_or_panic())
                .expect("fitness comparison failed")
        });
    }

    /// 从当前种群中克隆并提取适应度最高的前 `count` 个个体（精英个体）。
    ///
    /// 该方法通常用于“精英保留”策略（Elitism），确保当前代的最优解能直接进入下一代。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `count` - 需要保留的精英个体数量。如果 `count` 大于种群总数，则返回按适应度降序排列的整个种群。
    ///
    /// # 恐慌 (Panics)
    ///
    /// 依赖于适应度排序，如果存在未评估或适应度为 `NaN` 的个体，将会触发 panic。
    pub fn elite(&self, count: usize) -> Vec<Individual> {
        let mut sorted = self.individuals.clone();
        sorted.sort_by(|left, right| {
            right
                .fitness_or_panic()
                .partial_cmp(&left.fitness_or_panic())
                .expect("fitness comparison failed")
        });
        sorted.into_iter().take(count).collect()
    }

    /// 计算当前种群所有个体适应度的平均值。
    ///
    /// # 错误 (Errors)
    ///
    /// 如果种群为空（被零除），将返回 [`GaError::EmptyPopulation`]。
    ///
    /// # 恐慌 (Panics)
    ///
    /// 遍历计算过程中，如果遇到尚未计算适应度的个体，将会触发 panic。
    pub fn average_fitness(&self) -> Result<f64, GaError> {
        if self.is_empty() {
            return Err(GaError::EmptyPopulation);
        }

        let total = self
            .individuals
            .iter()
            .map(Individual::fitness_or_panic)
            .sum::<f64>();
        Ok(total / self.len() as f64)
    }

    /// 计算当前种群适应度的标准差。
    ///
    /// 该实现基于总体标准差（population standard deviation），
    /// 分母使用当前种群大小 `N` 而不是 `N - 1`。
    ///
    /// # Errors
    ///
    /// 如果种群为空，将返回 [`GaError::EmptyPopulation`]。
    ///
    /// # Panics
    ///
    /// 如果存在尚未评估适应度的个体，将在访问 fitness 时触发 panic。
    pub fn fitness_std_dev(&self) -> Result<f64, GaError> {
        let mean = self.average_fitness()?;
        let variance = self
            .individuals
            .iter()
            .map(|individual| {
                let delta = individual.fitness_or_panic() - mean;
                delta * delta
            })
            .sum::<f64>()
            / self.len() as f64;
        Ok(variance.sqrt())
    }
}
