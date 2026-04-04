use std::fmt::{Display, Formatter};

/// 遗传算法运行过程中可能发生的全部核心错误。
///
/// 该枚举集合了在配置校验、种群操作以及基因解析时可能出现的各种预期内（可恢复）的异常情况。
/// 通过返回 `GaError`，库可以优雅地将错误原因传递给调用者，避免程序不必要的崩溃。
#[derive(Debug, Clone, PartialEq)]
pub enum GaError {
    /// 无效配置错误。
    ///
    /// 当用户提供的 [`GaConfig`](crate::ga::config::GaConfig) 参数不符合逻辑要求
    /// （例如：种群规模为 0，交叉概率不在 `0.0~1.0` 之间，精英保留数大于种群总数等）时触发。
    /// 包含具体的文本信息以指导用户修正。
    InvalidConfig(String),

    /// 空种群错误。
    ///
    /// 当对一个尚未初始化或已被清空的种群执行必须有成员参与的操作时触发
    /// （例如：试图求取平均适应度，或提取最优个体）。
    EmptyPopulation,

    /// 适应度未评估错误。
    ///
    /// 当尝试访问或利用某个个体的适应度，但该个体尚未经过评估函数计算时触发。
    /// （注：部分内部逻辑为了性能可能会直接 panic，但此变体可用于对外暴露的安全接口）。
    UnevaluatedFitness,

    /// 不支持的基因类型错误。
    ///
    /// 当尝试将基因值转换为系统无法处理或未实现的标量类型（Scalar Type）时触发。
    /// 包含导致不支持的具体类型名称。
    UnsupportedGeneType(String),

    /// 可视化或报告生成失败。
    ///
    /// 用于封装图表绘制、目录创建和报告文件写入阶段的错误。
    Visualization(String),
}

/// 为 `GaError` 实现 `Display` trait，以提供终端友好的可读错误信息。
impl Display for GaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid config: {message}"),
            Self::EmptyPopulation => write!(f, "population is empty"),
            Self::UnevaluatedFitness => write!(f, "fitness is not evaluated"),
            Self::UnsupportedGeneType(name) => write!(f, "unsupported gene type: {name}"),
            Self::Visualization(message) => write!(f, "visualization error: {message}"),
        }
    }
}

/// 为 `GaError` 实现标准库的 `Error` trait。
///
/// 这使得 `GaError` 可以与 Rust 生态系统中的主流错误处理机制无缝对接，
/// 例如可以使用 `?` 向上抛出转化为 `Box<dyn std::error::Error>`，或与 `anyhow` 库配合使用。
impl std::error::Error for GaError {}
