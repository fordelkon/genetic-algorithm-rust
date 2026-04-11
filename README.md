# genetic-algorithm-rust

## 项目名称

genetic-algorithm-rust

## 项目介绍

genetic-algorithm-rust 是一个面向实验研究与工程原型的 Rust 遗传算法库。

它提供可组合的选择、交叉、变异和停止策略，支持并行适应度评估、可复现随机种子、以及可视化实验报告输出。项目目标是让你既能快速跑通 GA 工作流，也能在参数对比、实验复现和结果汇报时保持可追踪性。

## 功能特性

- Builder 风格配置：通过 `EngineConfig::builder(...)` 逐步构建实验参数
- 丰富算子支持：
  - 选择：`SteadyState`、`Tournament`、`RouletteWheel`、`Rank`、`StochasticUniversalSampling`
  - 交叉：`None`、`SinglePoint`、`TwoPoint`、`Uniform`
  - 变异：`None`、`RandomReset`、`RandomPerturbation`、`Adaptive*`、`Swap`、`Scramble`、`Inversion`
- 多基因类型：支持整数、无符号整数、浮点数（含混合逐基因类型）
- 搜索空间建模：支持全局域和逐基因域（离散/连续/步进）
- 并行评估：基于 Rayon 并行计算 fitness
- 岛屿模型：支持多岛并行进化与周期迁移
- 统计与报告：自动输出按代统计与图表（SVG + Markdown）

## 核心概念

### GA（Genetic Algorithm）

遗传算法通过“种群迭代”搜索解空间，核心步骤包括：

1. 初始化种群
2. 评估个体适应度
3. 选择父代
4. 交叉与变异生成新个体
5. 保留精英并进入下一代
6. 达到停止条件后输出最优解

在本库中，`EngineKernel` 即单种群 GA 执行核心。

### Island Model（岛屿模型）

岛屿模型将一个大种群拆分成多个子种群（岛），每个岛独立进化，并按固定间隔执行迁移：

- 优点：保持多样性、降低早熟收敛风险、可天然并行
- 关键参数：`num_islands`、`migration_count`、`migration_interval`、`migration_topology`

在本库中，`EvolutionEngine` 会根据 `num_islands` 自动选择单岛或岛屿后端。

## 安装方法

在 `Cargo.toml` 中添加依赖（Git 源）：

```toml
[dependencies]
genetic-algorithm-rust = { git = "https://github.com/fordelkon/genetic-algorithm-rust" }
```

在代码中导入：

```rust
use genetic_algorithm_rust::*;
```

## 快速开始（代码示例）

下面示例演示一个最小可运行 GA：让 8 个基因逼近目标值 2.0。

```rust
use genetic_algorithm_rust::{
    CrossoverType, EngineConfig, EngineKernel, GeneScalarType, GeneValue, GenesDomain,
    GenesValueType, MutationType, SelectionType, StopCondition,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig::builder(100, 8, 100, 12)
        .init_range(-4.0, 4.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .genes_domain(Some(GenesDomain::Global(
            genetic_algorithm_rust::GeneDomain::Continuous {
                low: -4.0,
                high: 4.0,
            },
        )))
        .selection_type(SelectionType::Tournament { k: 3 })
        .crossover(CrossoverType::SinglePoint, 0.9)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -0.5,
                max_delta: 0.5,
            },
            0.15,
        )
        .elitism_count(2)
        .random_seed(Some(42))
        .stop_condition(StopCondition::MaxGenerations)
        .build()?;

    let mut ga = EngineKernel::new(config, |genes| {
        let penalty = genes
            .iter()
            .map(|g| (g.to_f64() - 2.0).powi(2))
            .sum::<f64>();
        100.0 - penalty
    })?;

    ga.run()?;

    let best = ga.best_solution()?;
    println!("best fitness: {:.4}", best.fitness_or_panic());
    println!(
        "best genes: {:?}",
        best.genes.iter().map(GeneValue::to_f64).collect::<Vec<_>>()
    );

    ga.stats.render_report("output/quick-start")?;
    Ok(())
}
```

运行：

```bash
cargo run
```

## 配置说明

### 基础配置

- `population_size`: 种群规模
- `num_genes`: 每个个体基因数
- `num_generations`: 最大代数
- `num_parents_mating`: 每代参与交配的父代数

### 关键策略配置

- 选择策略：`selection_type(...)`
- 交叉策略与概率：`crossover(...)`
- 变异策略与概率：`mutation(...)`
- 精英保留：`elitism_count(...)`
- 随机种子：`random_seed(Some(seed))`
- 停止条件：`stop_condition(...)`

### 岛屿模型配置

通过 `island_model(...)` 启用：

```rust
use genetic_algorithm_rust::{EngineConfig, MigrationType};

let config = EngineConfig::builder(200, 16, 300, 24)
    .island_model(4, 5, 20, MigrationType::Ring)
    .build()?;
```

参数含义：

- `num_islands`: 岛数量（>1 时启用）
- `migration_count`: 每次迁移个体数
- `migration_interval`: 迁移间隔（代）
- `migration_topology`: 迁移拓扑（如 Ring）

## 架构设计

- `EngineConfig`：参数模型与校验入口
- `EngineKernel`：单种群演化执行器
- `IslandEngine`：多岛执行器（岛间迁移）
- `EvolutionEngine`：统一入口（自动分派单岛/多岛）
- `RunStats`：按代统计、摘要与报告导出

执行流（简化）：

1. 构建并校验配置
2. 初始化种群
3. 并行评估 fitness
4. 选择 -> 交叉 -> 变异 -> 精英保留
5. 更新统计并判断停止条件
6. 输出最优解与实验报告

## 示例

### 示例 1：统一入口（推荐）

```rust
use genetic_algorithm_rust::{EngineConfig, EvolutionEngine, GeneValue, StopCondition};

let config = EngineConfig::builder(120, 10, 150, 16)
    .stop_condition(StopCondition::MaxGenerations)
    .build()?;

let mut engine = EvolutionEngine::new(config, |genes| {
    -genes.iter().map(GeneValue::to_f64).map(f64::abs).sum::<f64>()
})?;

let _result = engine.run()?;
println!("best fitness: {:?}", engine.best_fitness());
```

### 示例 2：导出实验报告

```rust
ga.stats.render_report("output/experiment-a")?;
```

输出目录通常包含：

- `fitness_history.svg`
- `best_genes_final.svg`
- `best_genes_trajectory.svg`
- `summary.md`

## API 说明（简要）

常用公开类型：

- `EngineConfig` / `EngineConfigBuilder`
- `EngineKernel`
- `IslandEngine`
- `EvolutionEngine`
- `RunStats`
- `GeneValue` / `GenesDomain` / `GenesValueType`
- `SelectionType` / `CrossoverType` / `MutationType` / `StopCondition`

常用方法：

- `EngineConfig::builder(...)`
- `EngineKernel::new(config, fitness_fn)`
- `EvolutionEngine::new(config, fitness_fn)`
- `run()`
- `best_solution()` / `best_fitness()`
- `stats.summary()` / `stats.render_report(path)`

## 未来规划（可选）

- 多目标优化支持（如 Pareto-based workflow）
- 更丰富的迁移拓扑与异构岛策略
- 更细粒度的并行策略与性能调优开关
- 更完整的 benchmark 与案例集

## 贡献指南

欢迎提交 Issue 和 Pull Request。

建议流程：

1. Fork 本仓库并创建特性分支
2. 编写代码与测试
3. 运行质量检查：

```bash
cargo fmt
cargo test
```

4. 提交 PR，并说明变更动机、实现方案和测试结果

建议贡献内容：

- 新算子或新停止条件
- 文档改进与示例补充
- 性能优化与基准测试
- Bug 修复与回归测试
