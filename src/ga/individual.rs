use crate::ga::gene::GeneValue;

/// 遗传算法中的“个体”（Individual）。
///
/// 个体代表了问题搜索空间中的一个候选解（Candidate Solution）。
/// 它由一组基因（染色体，即解的具体参数）和该解的适应度（Fitness）组成。
///
/// 适应度被设计为 `Option<f64>` 类型，这是因为个体在刚刚被初始化，
/// 或者经历了交叉、变异等操作后，其内部的基因发生了改变，旧的适应度不再有效。
/// 此时 `fitness` 应为 `None`，直到经过评估函数（Evaluator）计算后才会重新被赋值。
#[derive(Debug, Clone, PartialEq)]
pub struct Individual {
    /// 个体的染色体（基因序列），代表一组待优化的变量。
    pub genes: Vec<GeneValue>,
    /// 个体的适应度评分。`None` 表示该个体尚未被评估或基因已发生更改。
    pub fitness: Option<f64>,
}

impl Individual {
    /// 构造一个新的个体实例。
    ///
    /// 默认情况下，新创建的个体尚未进行适应度评估，因此其 `fitness` 被初始化为 `None`。
    /// 该方法通常用于种群初始化，或者在交叉和变异操作中生成新子代时使用。
    pub fn new(genes: Vec<GeneValue>) -> Self {
        Self {
            genes,
            fitness: None,
        }
    }

    /// 构造一个带有已知适应度的新个体实例。
    ///
    /// 此方法直接为个体赋予一个评估好的适应度。
    /// 通常用于测试场景、手动注入已知优质解，或者在拷贝已经评估过的个体时使用。
    pub fn with_fitness(genes: Vec<GeneValue>, fitness: f64) -> Self {
        Self {
            genes,
            fitness: Some(fitness),
        }
    }

    /// 提取并返回当前个体的适应度值。
    ///
    /// 该方法不会执行任何复杂的数学计算，它只是简单地从 `Option` 中取出适应度的数值。
    ///
    /// # 恐慌 (Panics)
    ///
    /// 此方法假定调用时，个体已经完成了适应度评估。
    /// 如果当前个体的 `fitness` 为 `None`（例如刚发生过变异但还未重新评估），
    /// 调用此方法将会直接触发 **panic**，并输出 `"individual fitness should be evaluated"`。
    pub fn fitness_or_panic(&self) -> f64 {
        self.fitness
            .expect("individual fitness should be evaluated")
    }
}
