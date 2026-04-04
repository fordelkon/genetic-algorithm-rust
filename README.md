# genetic-algorithm-rust

一个面向实验与工程使用场景的 Rust 遗传算法库。

它提供了可配置的选择、交叉、变异与停止条件，支持并行 fitness 评估、可复现随机种子，以及运行完成后的统计摘要与可视化实验报告输出。对于第一次使用该库的开发者，它强调“能快速跑起来”；对于课程实验或研究复现场景，它强调“可解释、可追踪、可导出”。

## Why This Library

这个库适合你在以下场景中使用：

- 做参数搜索、数值优化或启发式求解
- 作为教学或课程设计中的遗传算法实验框架
- 需要复现同一组实验结果并导出图表与摘要
- 想用 Rust 获得较好的性能和更严格的类型约束

相较于“只能跑出一个最优解”的最小实现，它还会自动记录按代统计信息，并生成一套可直接用于汇报或分析的实验报告文件。

## Features

- Builder 风格配置：通过 `GaConfig::builder(...)` 逐步构造参数
- 支持多种父代选择策略
  - `SteadyState`
  - `Tournament { k }`
  - `RouletteWheel`
  - `Rank`
  - `StochasticUniversalSampling`
- 支持多种交叉策略
  - `None`
  - `SinglePoint`
  - `TwoPoint`
  - `Uniform`
- 支持多种变异策略
  - `None`
  - `RandomReset`
  - `RandomPerturbation`
  - `AdaptiveRandomReset`
  - `AdaptiveRandomPerturbation`
  - `Swap`
  - `Scramble`
  - `Inversion`
- 支持多种基因数值类型
  - `i8/i16/i32/i64/isize`
  - `u8/u16/u32/u64/usize`
  - `f32/f64`
- 支持统一或逐基因定义搜索空间
  - `Discrete`
  - `Continuous`
  - `Stepped`
- 支持多种停止条件
  - `MaxGenerations`
  - `TargetFitness`
  - `NoImprovement`
- 并行 fitness 评估
  - `evaluate_population()` 内部基于 Rayon
  - `fitness_fn` 需满足 `Fn(&[GeneValue]) -> f64 + Sync`
- 可复现实验结果
  - `random_seed`
- 自动统计运行过程
  - 每代最优适应度
  - 每代平均适应度
  - 每代适应度标准差
  - 每代最优个体基因快照
  - 全局最优解与全局最优适应度
- 原生实验报告输出
  - `fitness_history.svg`
  - `best_genes_final.svg`
  - `best_genes_trajectory.svg`
  - `summary.md`

## Installation

在 `Cargo.toml` 中添加：

```toml
[dependencies]
genetic-algorithm-rust = "0.1.0"
```

在代码中导入时，crate 名称使用下划线形式：

```rust
use genetic_algorithm_rust::*;
```

## Quick Start

下面这个例子会让 8 个基因尽量逼近目标值 `2.0`，并在运行结束后自动输出一份实验报告。

```rust
use genetic_algorithm_rust::{
    CrossoverType, GaConfig, GeneDomain, GeneScalarType, GeneValue, GenesDomain, GenesValueType,
    GeneticAlgorithm, MutationType, SelectionType, StopCondition,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = GaConfig::builder(100, 8, 100, 12)
        .init_range(-4.0, 4.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .genes_domain(Some(GenesDomain::Global(GeneDomain::Continuous {
            low: -4.0,
            high: 4.0,
        })))
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

    let mut ga = GeneticAlgorithm::new(config, |genes| {
        let penalty = genes
            .iter()
            .map(|gene| (gene.to_f64() - 2.0).powi(2))
            .sum::<f64>();

        100.0 - penalty
    })?;

    let stats = ga.run()?;
    let best = ga.best_solution()?;

    println!("Best fitness: {:?}", best.fitness);
    println!(
        "Best genes: {:?}",
        best.genes.iter().map(GeneValue::to_f64).collect::<Vec<_>>()
    );

    ga.stats.render_report("output/example-run")?;

    println!("Summary: {:?}", stats.summary()?);

    Ok(())
}
```

运行后，你会同时得到：

- 一个可用于程序逻辑的 `RunStats`
- 当前种群最优个体 `best_solution()`
- 一个包含图表与摘要文件的实验输出目录

## Generated Report

调用：

```rust
ga.stats.render_report("output/example-run")?;
```

默认会生成：

```text
output/example-run/
├── fitness_history.svg
├── best_genes_final.svg
├── best_genes_trajectory.svg
└── summary.md
```

各文件含义如下：

- `fitness_history.svg`
  - 展示每代 `best fitness`
  - 展示每代 `average fitness`
  - 展示平滑后的 `Smoothed Average`
  - 展示基于 `avg ± std` 的离散度阴影带
- `best_genes_final.svg`
  - 展示最终全局最优个体各基因值
- `best_genes_trajectory.svg`
  - 使用单页 small multiples 分面展示每个基因的演化轨迹
  - 如果基因数量超过网格容量，最后一个分面会显示省略提示卡
- `summary.md`
  - 记录代数、最优值、末代平均值、末代标准差、提升量和最优基因向量

这套输出适合：

- 调试收敛行为
- 做课程实验截图
- 记录不同参数组合的实验结果
- 生成可复现的阶段性报告

## Core Workflow

一次完整使用通常包含 5 步。

### 1. 定义配置

主入口是：

```rust
GaConfig::builder(population_size, num_genes, num_generations, num_parents_mating)
```

你可以在 builder 上继续配置：

- 初始范围
- 基因类型
- 搜索空间
- 父代选择策略
- 交叉策略与概率
- 变异策略与概率
- 精英保留
- 停止条件
- 随机种子

### 2. 定义 fitness 函数

适应度函数签名：

```rust
Fn(&[GeneValue]) -> f64 + Sync
```

返回值越大表示个体越优。

由于 `evaluate_population()` 与 `run()` 内部会并行评估种群，因此闭包需要满足 `Sync`。

常见写法是先把 `GeneValue` 转成 `f64`：

```rust
let fitness = |genes: &[GeneValue]| {
    genes.iter().map(GeneValue::to_f64).sum::<f64>()
};
```

### 3. 运行算法

```rust
let mut ga = GeneticAlgorithm::new(config, fitness)?;
let stats = ga.run()?;
```

`run()` 会完成：

- 初代种群初始化
- 全种群适应度评估
- 选择、交叉、变异、精英保留
- 停止条件判断
- 统计记录

### 4. 读取最优解与统计信息

```rust
let best = ga.best_solution()?;
let summary = stats.summary()?;
```

你可以直接读取：

- 当前最优个体
- 全局最优适应度
- 历史 fitness 曲线
- 摘要统计值

### 5. 导出实验报告

```rust
ga.stats.render_report("output/my-experiment")?;
```

这是最适合第一次使用者上手的路径：跑一次实验，直接查看图与摘要，再决定下一步怎么调参。

## Configuration Guide

`GaConfig::builder(...)` 的默认配置如下：

- `init_range`: `-4.0..=4.0`
- `genes_value_type`: 所有基因默认 `F64`
- `selection_type`: `Tournament { k: 3 }`
- `crossover`: `SinglePoint`, 概率 `0.8`
- `mutation`: `RandomPerturbation`, 概率 `0.05`
- `mutation_num_genes`: `None`
- `elitism_count`: `1`
- `stop_condition`: `MaxGenerations`

如果你刚开始使用，建议优先只改以下几个参数：

- `population_size`
- `num_generations`
- `selection_type`
- `mutation(...)`
- `elitism_count`
- `random_seed`

## Selection, Crossover, and Mutation

### Selection

当前支持：

- `SelectionType::SteadyState`
- `SelectionType::Tournament { k }`
- `SelectionType::RouletteWheel`
- `SelectionType::Rank`
- `SelectionType::StochasticUniversalSampling`

说明：

- `SteadyState` 直接按 fitness 取前若干个体，选择压力高
- `Tournament { k }` 是通用性最强的默认选择器
- `RouletteWheel` 按 fitness 比例采样；若 fitness 含负值，会自动平移
- `Rank` 按排名分配权重，适合缓解极端 fitness 主导
- `StochasticUniversalSampling` 比普通轮盘赌方差更小

### Crossover

当前支持：

- `CrossoverType::None`
- `CrossoverType::SinglePoint`
- `CrossoverType::TwoPoint`
- `CrossoverType::Uniform`

### Mutation

当前支持：

- `MutationType::None`
- `MutationType::RandomReset { min, max }`
- `MutationType::RandomPerturbation { min_delta, max_delta }`
- `MutationType::AdaptiveRandomReset { min, max, low_quality_num_genes, high_quality_num_genes }`
- `MutationType::AdaptiveRandomPerturbation { min_delta, max_delta, low_quality_num_genes, high_quality_num_genes }`
- `MutationType::Swap`
- `MutationType::Scramble`
- `MutationType::Inversion`

说明：

- `RandomReset` / `RandomPerturbation` 可按 `mutation_probability` 逐基因触发
- 也可通过 `mutation_num_genes(k)` 改为“每个个体固定选择 `k` 个基因变异”
- 自适应变异会根据个体 fitness 相对群体均值动态调整变异强度
- `Swap` / `Scramble` / `Inversion` 更适合顺序编码类问题

固定变异基因数示例：

```rust
use genetic_algorithm_rust::{GaConfig, MutationType};

let config = GaConfig::builder(50, 8, 100, 10)
    .mutation(
        MutationType::RandomReset {
            min: -5.0,
            max: 5.0,
        },
        0.0,
    )
    .mutation_num_genes(2)
    .build()?;
```

自适应变异示例：

```rust
use genetic_algorithm_rust::{GaConfig, MutationType};

let config = GaConfig::builder(50, 8, 100, 10)
    .mutation(
        MutationType::AdaptiveRandomPerturbation {
            min_delta: -0.2,
            max_delta: 0.2,
            low_quality_num_genes: 3,
            high_quality_num_genes: 1,
        },
        1.0,
    )
    .build()?;
```

## Gene Types and Search Spaces

### Gene Scalar Types

支持的数值类型：

- `i8/i16/i32/i64/isize`
- `u8/u16/u32/u64/usize`
- `f32/f64`

如果需要混合类型，可以使用逐基因定义：

```rust
GenesValueType::PerGene(vec![
    GeneScalarType::I8,
    GeneScalarType::U16,
    GeneScalarType::F32,
    GeneScalarType::F64,
])
```

### Search Spaces

支持两种空间组织方式：

- `GenesDomain::Global(...)`
- `GenesDomain::PerGene(...)`

支持的单基因空间：

- `GeneDomain::Discrete(Vec<f64>)`
- `GeneDomain::Continuous { low, high }`
- `GeneDomain::Stepped { low, high, step }`

这使你可以表达：

- 全部基因共享一个连续搜索区间
- 某些基因只能取有限离散值
- 某些基因只能按固定步长取值
- 每个基因有不同的数值类型与搜索空间

## Reproducibility and Stopping

### Reproducibility

通过：

```rust
.random_seed(Some(42))
```

你可以让初始化、选择、交叉与变异过程可复现。这对以下场景很重要：

- 调试问题
- 比较不同参数设置
- 复现实验报告
- 课程或论文附录说明

### Stop Conditions

当前支持：

- `StopCondition::MaxGenerations`
- `StopCondition::TargetFitness(...)`
- `StopCondition::NoImprovement { generations, min_delta }`

如果你只是先把流程跑通，建议先用 `MaxGenerations`。等目标更明确后，再切换到基于 fitness 的停止条件。

## Statistics and Experiment Outputs

运行完成后，`RunStats` 会保存整个实验过程的关键信息：

- `best_fitness_per_generation`
- `avg_fitness_per_generation`
- `std_fitness_per_generation`
- `best_genes_per_generation`
- `best_solution`
- `best_fitness`

其中：

- `best_fitness_per_generation` 用于看收敛上界
- `avg_fitness_per_generation` 用于看群体整体趋势
- `std_fitness_per_generation` 用于看群体是否逐渐失去多样性
- `best_genes_per_generation` 用于绘制基因轨迹分面图

同时，你还可以通过：

```rust
let summary = stats.summary()?;
```

获得 `ExperimentSummary`，其中包括：

- 总代数
- 全局最优适应度
- 末代平均适应度
- 末代标准差
- 初代最优与末代最优
- 提升量
- 最优基因向量

## Public Types You Will Commonly Use

初次使用时，最常见的公开类型有：

- `GaConfig`
- `GeneticAlgorithm`
- `GeneValue`
- `SelectionType`
- `CrossoverType`
- `MutationType`
- `StopCondition`
- `ExperimentSummary`
- `VisualizationOptions`

如果你只是想快速上手，优先掌握：

- `GaConfig::builder(...)`
- `GeneticAlgorithm::new(...)`
- `ga.run()`
- `ga.best_solution()`
- `ga.stats.summary()`
- `ga.stats.render_report(...)`

## Use Cases

这个库目前尤其适合：

- 面向开发者的原型优化实验
- 课程作业中的 GA 实现与可视化展示
- 参数调优过程中的策略比较
- 需要保存图表和摘要的重复实验

## Notes

当前版本更偏向单目标优化工作流：

- fitness 越大越优
- 运行统计与报告输出默认围绕单目标实验组织

后续可考虑：

- 多次独立运行的对比报告
- CSV / JSON 实验数据导出
- 更多报告主题与图表风格
- 更细粒度的自定义可视化配置

## 参考仓库

- [PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython)
