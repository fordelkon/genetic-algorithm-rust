use crate::ga::analysis::stats::RunStats;
use crate::ga::error::GaError;

/// 单次遗传算法实验的关键摘要指标。
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSummary {
    /// 实际记录到的代数数量。
    ///
    /// 该值等于所有代际历史向量的长度。
    pub generations: usize,
    /// 整个运行过程中观测到的全局最优适应度。
    pub best_fitness: f64,
    /// 最后一代种群的平均适应度。
    pub final_avg_fitness: f64,
    /// 最后一代种群适应度的标准差。
    ///
    /// 可用于衡量算法收敛末期种群内部的离散程度。
    pub final_std_fitness: f64,
    /// 初始代中的最优适应度。
    pub initial_best_fitness: f64,
    /// 最后一代中的最优适应度。
    pub final_best_fitness: f64,
    /// 末代最优值相对初代最优值的提升量。
    pub improvement: f64,
    /// 全局最优个体的基因值，统一转换为 `f64` 便于展示与导出。
    pub best_genes: Vec<f64>,
}

impl ExperimentSummary {
    /// 从一次完整运行的 [`RunStats`] 中提取实验摘要。
    ///
    /// # Errors
    ///
    /// 当统计历史为空，或者缺少生成摘要所需的全局最优解/适应度信息时，
    /// 返回 [`GaError::Visualization`]。
    pub fn from_stats(stats: &RunStats) -> Result<Self, GaError> {
        let generations = stats.best_fitness_per_generation.len();
        if generations == 0 {
            return Err(GaError::Visualization(
                "cannot summarize an empty run history".into(),
            ));
        }

        let initial_best_fitness = stats.best_fitness_per_generation[0];
        let final_best_fitness = *stats
            .best_fitness_per_generation
            .last()
            .expect("non-empty history should have last item");
        let final_avg_fitness = *stats
            .avg_fitness_per_generation
            .last()
            .ok_or_else(|| GaError::Visualization("average fitness history is empty".into()))?;
        let best_fitness = stats
            .best_fitness
            .ok_or_else(|| GaError::Visualization("global best fitness is missing".into()))?;
        let final_std_fitness = *stats
            .std_fitness_per_generation
            .last()
            .ok_or_else(|| GaError::Visualization("fitness std-dev history is empty".into()))?;
        let best_genes = stats
            .best_solution
            .as_ref()
            .ok_or_else(|| GaError::Visualization("global best solution is missing".into()))?
            .iter()
            .map(|gene| gene.to_f64())
            .collect();

        Ok(Self {
            generations,
            best_fitness,
            final_avg_fitness,
            final_std_fitness,
            initial_best_fitness,
            final_best_fitness,
            improvement: final_best_fitness - initial_best_fitness,
            best_genes,
        })
    }
}
