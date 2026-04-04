use std::path::Path;

use crate::ga::{
    error::GaError, gene::GeneValue, population::Population, report::ExperimentSummary, visualize,
};

/// 遗传算法运行过程的统计与历史记录器。
///
/// `RunStats` 用于追踪算法在每一代（Generation）演化过程中的收敛趋势，
/// 同时负责拦截并持久化保存整个生命周期中出现过的**全局最优解**。
/// 这部分数据非常适合用于最终的实验结果输出或可视化图表的绘制。
#[derive(Debug, Clone, Default)]
pub struct RunStats {
    /// 记录每一代种群中的最优个体适应度。
    ///
    /// 数组的索引即代表代数（Generation）。通常用于绘制算法的“收敛曲线”。
    pub best_fitness_per_generation: Vec<f64>,

    /// 记录每一代种群的平均适应度。
    ///
    /// 结合最优适应度曲线，可以判断种群是否失去了多样性（早熟收敛）。
    pub avg_fitness_per_generation: Vec<f64>,

    /// 记录每一代种群适应度的标准差。
    ///
    /// 该指标用于可视化群体离散程度和探索收缩过程。
    pub std_fitness_per_generation: Vec<f64>,

    /// 记录每一代最优个体的基因序列。
    ///
    /// 这组快照可用于绘制最优个体的基因演化轨迹图。
    pub best_genes_per_generation: Vec<Vec<GeneValue>>,

    /// 算法运行至今发现的**全局最优**个体的基因序列（染色体）。
    ///
    /// 若尚未进行任何记录，则为 `None`。
    pub best_solution: Option<Vec<GeneValue>>,

    /// 算法运行至今发现的**全局最高**适应度评分。
    ///
    /// 若尚未进行任何记录，则为 `None`。
    pub best_fitness: Option<f64>,
}

impl RunStats {
    /// 从给定的种群中提取统计指标并记录到历史档案中。
    ///
    /// 每次调用此方法，都会向历史数组追加当前代的最优和平均适应度。
    /// 同时，它会检查当前代的最优个体是否超越了历史全局最优。如果超越，则覆盖更新全局最优解。
    ///
    /// # 参数 (Arguments)
    ///
    /// * `population` - 刚刚完成适应度评估的当前代种群。
    ///
    /// # 错误 (Errors)
    ///
    /// 如果传入的种群为空（无任何个体可供评估），将返回 [`GaError::EmptyPopulation`]。
    ///
    /// # 恐慌 (Panics)
    ///
    /// 该方法假定传入的 `population` 中所有个体的适应度均已评估完毕。
    /// 如果存在未评估（`fitness` 为 `None`）或适应度为 `NaN` 的个体，将会触发 panic。
    pub fn record(&mut self, population: &Population) -> Result<(), GaError> {
        let best = population.best()?;
        let avg = population.average_fitness()?;
        let std_dev = population.fitness_std_dev()?;

        // 记录本代的统计数据
        self.best_fitness_per_generation
            .push(best.fitness_or_panic());
        self.avg_fitness_per_generation.push(avg);
        self.std_fitness_per_generation.push(std_dev);
        self.best_genes_per_generation.push(best.genes.clone());

        // 如果历史为空，或者当前代的最优个体比历史最优还要好，则刷新全局记录
        if self
            .best_fitness
            .is_none_or(|fitness| best.fitness_or_panic() > fitness)
        {
            self.best_fitness = best.fitness;
            self.best_solution = Some(best.genes.clone());
        }

        Ok(())
    }

    /// 获取最后一次记录（通常是最近一代）的最优适应度。
    ///
    /// 如果尚未记录过任何数据，则返回 `None`。
    pub fn last_best(&self) -> Option<f64> {
        self.best_fitness_per_generation.last().copied()
    }

    /// 计算一次实验运行的摘要指标。
    ///
    /// 返回值适合用于终端摘要、Markdown 报告或后续实验对比输出。
    ///
    /// # Errors
    ///
    /// 当历史记录为空，或者缺少全局最优解等必要信息时，返回 [`GaError::Visualization`]。
    pub fn summary(&self) -> Result<ExperimentSummary, GaError> {
        ExperimentSummary::from_stats(self)
    }

    /// 将运行历史渲染为完整实验报告。
    ///
    /// 默认会在目标目录下生成：
    /// - `fitness_history.svg`
    /// - `best_genes_final.svg`
    /// - `best_genes_trajectory.svg`
    /// - `summary.md`
    ///
    /// 图像样式由 [`visualize::VisualizationOptions::default`] 控制。
    ///
    /// # Errors
    ///
    /// 当历史记录为空、目录无法创建或图表文件写入失败时，返回对应错误。
    pub fn render_report<P: AsRef<Path>>(&self, output_dir: P) -> Result<(), GaError> {
        visualize::render_report(
            self,
            output_dir.as_ref(),
            &visualize::VisualizationOptions::default(),
        )
    }
}
