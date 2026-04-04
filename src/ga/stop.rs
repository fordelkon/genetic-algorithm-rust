use crate::ga::stats::RunStats;

/// 遗传算法的终止条件策略。
///
/// `StopCondition` 定义了算法在演化过程中何时应该停止迭代。
/// 支持通过最大迭代次数、达到目标适应度、以及适应度长期停滞（若干代无提升）等方式，
/// 灵活控制算法的生命周期，避免无效的计算资源浪费。
#[derive(Debug, Clone)]
pub enum StopCondition {
    /// 达到指定的最大迭代次数后终止。
    ///
    /// 这是最基础的终止条件，直接依赖于外部传入的 `max_generations` 参数。
    MaxGenerations,

    /// 当种群的最优适应度达到或超过预期的目标值时终止。
    ///
    /// 包含一个 `f64` 类型的值，表示期望达到的目标适应度阈值。
    TargetFitness(f64),

    /// 当种群的最优适应度在连续若干代内没有实质性提升（陷入停滞或早熟收敛）时终止。
    ///
    /// 包含一个 `generations` 字段，表示允许适应度不提升的最大连续代数。
    NoImprovement { generations: usize },
}

/// 评估当前演化状态是否满足预设的算法终止条件。
///
/// 该函数会根据选定的 `StopCondition` 策略，结合当前的迭代代数、最优适应度以及历史统计数据，
/// 来决定遗传算法是否应当结束运行。
/// **注意**：无论选用哪种策略，`max_generations` 都会作为硬性兜底条件。
///
/// # 参数 (Arguments)
///
/// * `condition` - 当前选用的终止条件策略。
/// * `current_generation` - 算法当前已经执行的代数。
/// * `best_fitness` - 当前代中评估出的最优个体适应度值。
/// * `stats` - 算法运行至今的统计与历史记录器。用于提供过往的适应度追踪记录，以判定是否陷入停滞。
/// * `max_generations` - 允许运行的绝对最大迭代次数（硬性上限兜底）。
///
/// # 返回值 (Returns)
///
/// 如果满足终止条件（或已达到最大迭代上限），返回 `true` 表示算法应当停止；否则返回 `false`。
///
/// # 恐慌 (Panics)
///
/// 当选用 `StopCondition::NoImprovement` 策略，且满足了历史记录长度的条件，
/// 但 `stats.best_fitness_per_generation` 实际上为空时，将会触发 panic。
/// （在正常算法流程中，由于先记录再判断，历史记录不应为空）。
pub fn should_stop(
    condition: &StopCondition,
    current_generation: usize,
    best_fitness: f64,
    stats: &RunStats,
    max_generations: usize,
) -> bool {
    match condition {
        StopCondition::MaxGenerations => current_generation >= max_generations,
        StopCondition::TargetFitness(target) => {
            current_generation >= max_generations || best_fitness >= *target
        }
        StopCondition::NoImprovement { generations } => {
            // 兜底：如果达到最大代数，直接停止
            if current_generation >= max_generations {
                return true;
            }

            let history = &stats.best_fitness_per_generation;

            // 如果历史记录的长度尚不足以用于比对，则继续迭代
            if history.len() <= *generations {
                return false;
            }

            let last = *history.last().expect("history should not be empty");

            // 提取过去 `generations` 代的历史记录切片
            let stagnant_slice = &history[history.len() - 1 - generations..history.len() - 1];

            // 检查切片中的历史最佳适应度是否都大于或等于当前代的最优适应度
            // 如果全部满足，说明在这连续 `generations` 代中算法没有取得任何实质性进展
            stagnant_slice.iter().all(|fitness| *fitness >= last)
        }
    }
}
