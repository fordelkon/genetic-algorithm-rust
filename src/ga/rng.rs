use rand::{SeedableRng, rngs::StdRng};

/// 构造并初始化一个标准随机数生成器（`StdRng`）。
///
/// 在遗传算法中，种群初始化、交叉配对、基因变异等核心算子都高度依赖于随机性。
/// 该函数为整个算法提供了一个统一的随机数引擎。
///
/// # 参数 (Arguments)
///
/// * `seed` - 可选的 64 位无符号整数种子。
///   - 若传入 `Some(value)`，将使用固定的种子进行初始化。这意味着在相同的全局配置下，
///     算法的每一步随机行为都是**确定且完全可复现**的，非常适合用于单元测试、Bug 排查和科学实验。
///   - 若传入 `None`，函数将调用操作系统/环境的随机熵源（通过 `rand::random()`）
///     生成一个真随机种子。每次运行算法都会得到完全不同的演化轨迹。
///
/// # 返回 (Returns)
///
/// 返回一个基于密码学安全（或伪随机高质量）的标准随机数生成器实例 [`StdRng`]。
pub fn build_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(value) => StdRng::seed_from_u64(value),
        None => StdRng::seed_from_u64(rand::random()),
    }
}
