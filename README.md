# genetic-algorithm-rust

## 项目名称

genetic-algorithm-rust

## 项目介绍

genetic-algorithm-rust 是一个面向实验研究与工程原型的 Rust 遗传算法库。

它提供可组合的选择、交叉、变异和停止策略，支持单目标 GA 与多目标 NSGA-II、并行适应度评估、可复现随机种子、以及可视化实验报告输出。项目目标是让你既能快速跑通 GA 工作流，也能在参数对比、实验复现和结果汇报时保持可追踪性。

## 功能特性

- Builder 风格配置：通过 `EngineConfig::builder(...)` 逐步构建实验参数
- 双优化模式：支持单目标最大化与 `OptimizationMode::Nsga2 { num_objectives }` 多目标优化
- 丰富算子支持：
  - 选择：`SteadyState`、`Tournament`、`RouletteWheel`、`Rank`、`StochasticUniversalSampling`
  - 交叉：`None`、`SinglePoint`、`TwoPoint`、`Uniform`
  - 变异：`None`、`RandomReset`、`RandomPerturbation`、`Adaptive*`、`Swap`、`Scramble`、`Inversion`
- 多基因类型：支持整数、无符号整数、浮点数（含混合逐基因类型）
- 搜索空间建模：支持全局域和逐基因域（离散/连续/步进）
- 统一评估接口：单目标使用标量 fitness，多目标使用 `Evaluation::Multi(Vec<f64>)`
- 并行评估：基于 Rayon 并行计算 fitness / objectives
- 岛屿模型：支持多岛并行进化与周期迁移
- Pareto 前沿导出：支持通过 `pareto_front()` 获取非支配解集和 `ParetoSolution`
- 统计与报告：自动输出按代统计与图表（SVG + Markdown），包含 NSGA-II 的 Pareto 前沿报告

## 核心概念

### GA（Genetic Algorithm）

遗传算法通过“种群迭代”搜索解空间，核心步骤包括：

1. 初始化种群
2. 评估个体适应度
3. 选择父代
4. 交叉与变异生成新个体
5. 保留精英并进入下一代
6. 达到停止条件后输出最优解

### Island Model（岛屿模型）

岛屿模型将一个大种群拆分成多个子种群（岛），每个岛独立进化，并按固定间隔执行迁移：

- 优点：保持多样性、降低早熟收敛风险、可天然并行
- 关键参数：`num_islands`、`migration_count`、`migration_interval`、`migration_topology`

在本库中，`EvolutionEngine` 会根据 `num_islands` 自动选择单岛或岛屿后端。

### NSGA-II / Pareto Front

当问题存在多个相互冲突的目标时，可以将引擎切换到 `OptimizationMode::Nsga2 { num_objectives }`：

- evaluator 返回 `Vec<f64>`，库会自动封装为 `Evaluation::Multi(...)`
- 目标按“最小化”解释，内部使用非支配排序与 crowding distance 维持解集多样性
- 单目标模式下常用的 `best_fitness()` / `best_solution()` 在 NSGA-II 模式下不可用，应改用 `pareto_front()`
- `pareto_front()` 返回 `Vec<ParetoSolution>`，其中包含 `genes`、`objectives`、`rank`、`crowding_distance`

这使得该库既能处理传统标量 fitness 优化，也能直接用于 Pareto 最优解集搜索与可视化分析。

## 安装方法

在 `Cargo.toml` 中添加依赖（Git 源）：

```toml
[dependencies]
genetic-algorithm-rust = { git = "https://github.com/fordelkon/genetic-algorithm-rust" }
```

## 快速开始（代码示例）

### 示例 1：基准函数（连续优化）

```rust
use genetic_algorithm_rust::{
    CrossoverType, EngineConfig, GeneDomain, GeneScalarType, GenesDomain,
    GenesValueType, EvolutionEngine, MigrationType, MutationType, SelectionType, StopCondition,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig::builder(100, 10, 1000, 20)
        .init_range(-5.12, 5.12)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .genes_domain(Some(GenesDomain::Global(
            GeneDomain::Continuous {
                low: -5.12,
                high: 5.12,
            },
        )))
        .island_model(4, 5, 20, MigrationType::Ring)
        .crossover(CrossoverType::SinglePoint, 0.8)
        .mutation(
            MutationType::AdaptiveRandomPerturbation {
                min_delta: -0.2,
                max_delta: 0.2,
                low_quality_num_genes: 5,
                high_quality_num_genes: 1,
            },
            0.15,
        )
        .elitism_count(5)
        .selection_type(SelectionType::Tournament { k: 3 })
        .random_seed(Some(42))
        .stop_condition(StopCondition::Any {
            target_fitness: None,
            no_improvement_generations: Some(100),
        })
        .build()?;

    let mut ga = EvolutionEngine::new(config, |genes| {
        // Ackley: 经典多峰函数，常用于测试全局优化能力
        let xs: Vec<f64> = genes.iter().map(|g| g.to_f64()).collect();
        let n = xs.len() as f64;
        let sum_sq = xs.iter().map(|x| x.powi(2)).sum::<f64>();
        let sum_cos = xs
            .iter()
            .map(|x| (2.0 * std::f64::consts::PI * x).cos())
            .sum::<f64>();

        let objective = -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp()
            - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E;

        // 最小化 objective -> 最大化 fitness
        -objective
    })?;

    ga.run()?;

    let (best_island, best_fitness) = ga
        .best_fitness()
        .expect("best fitness missing");
    let (_, best) = ga.best_solution().expect("best solution missing");
    let best_genes = best
        .genes
        .iter()
        .map(|g| g.to_f64())
        .collect::<Vec<_>>();

    println!("best island: {best_island}, best fitness: {best_fitness:.6}");
    println!("best solution genes: {:?}", best_genes);
    ga.stats.render_report("output/multiple-populations")?;
    Ok(())
}
```

### 示例 2：TSP（离散编码）

```rust
use genetic_algorithm_rust::{
    CrossoverType, EngineConfig, GeneDomain, GeneScalarType, GeneValue, GenesDomain,
    GenesValueType, EvolutionEngine, MigrationType, MutationType, SelectionType, StopCondition,
};

const DIST: [[f64; 5]; 5] = [
    [0.0, 2.0, 9.0, 10.0, 7.0],
    [2.0, 0.0, 6.0, 4.0, 3.0],
    [9.0, 6.0, 0.0, 8.0, 5.0],
    [10.0, 4.0, 8.0, 0.0, 6.0],
    [7.0, 3.0, 5.0, 6.0, 0.0],
];

fn repair_permutation(raw: &[usize], n: usize) -> Vec<usize> {
    let mut seen = vec![false; n];
    let mut repaired = Vec::with_capacity(n);

    for &x in raw {
        if x < n && !seen[x] {
            seen[x] = true;
            repaired.push(x);
        }
    }

    for city in 0..n {
        if !seen[city] {
            repaired.push(city);
        }
    }

    repaired
}

fn decode_tsp_route(genes: &[GeneValue]) -> Vec<usize> {
    let raw: Vec<usize> = genes
        .iter()
        .map(|g| match g {
            GeneValue::I32(city) => usize::try_from(*city).expect("city must be non-negative"),
            _ => panic!("TSP example expects GeneValue::I32"),
        })
        .collect();
    repair_permutation(&raw, 5)
}

fn tsp_tour_length(route: &[usize]) -> f64 {
    let n = route.len();
    let mut total = 0.0;
    for i in 0..n {
        let from = route[i];
        let to = route[(i + 1) % n];
        total += DIST[from][to];
    }
    total
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let city_domain = GeneDomain::Discrete((0..5).map(|x| x as f64).collect());

    let config = EngineConfig::builder(80, 5, 200, 16)
        .init_range(0.0, 4.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::I32))
        .genes_domain(Some(GenesDomain::Global(city_domain)))
        .island_model(4, 2, 12, MigrationType::Ring)
        .crossover(CrossoverType::SinglePoint, 0.9)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -1.0,
                max_delta: 1.0,
            },
            0.2,
        )
        .elitism_count(2)
        .selection_type(SelectionType::Tournament { k: 3 })
        .random_seed(Some(42))
        .stop_condition(StopCondition::Any {
            target_fitness: None,
            no_improvement_generations: Some(40),
        })
        .build()?;

    let mut ga = EvolutionEngine::new(config, |genes| -tsp_tour_length(&decode_tsp_route(genes)))?;

    ga.run()?;

    let (best_island, best) = ga.best_solution()?;
    let best_route = decode_tsp_route(&best.genes);
    let best_length = tsp_tour_length(&best_route);

    println!("best island: {best_island}");
    println!("best fitness: {:.4}", best.fitness_or_panic());
    println!("best route: {:?}", best_route);
    println!("best tour length: {:.4}", best_length);
    ga.stats.render_report("output/tsp")?;
    Ok(())
}
```

### 示例 3：NSGA-II（多目标优化）

```rust
use genetic_algorithm_rust::{
    CrossoverType, EngineConfig, EvolutionEngine, GeneDomain, GeneScalarType, GeneValue,
    GenesDomain, GenesValueType, MutationType, OptimizationMode, StopCondition,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig::builder(48, 2, 80, 16)
        .init_range(0.0, 1.0)
        .genes_value_type(GenesValueType::All(GeneScalarType::F64))
        .genes_domain(Some(GenesDomain::Global(GeneDomain::Continuous {
            low: 0.0,
            high: 1.0,
        })))
        .crossover(CrossoverType::SinglePoint, 0.85)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -0.15,
                max_delta: 0.15,
            },
            0.2,
        )
        .optimization_mode(OptimizationMode::Nsga2 { num_objectives: 2 })
        .stop_condition(StopCondition::MaxGenerations)
        .random_seed(Some(19))
        .build()?;

    let mut engine = EvolutionEngine::new(config, |genes: &[GeneValue]| {
        let x1 = genes[0].to_f64();
        let x2 = genes[1].to_f64();
        let f1 = x1;
        let g = 1.0 + 9.0 * x2;
        let f2 = g * (1.0 - (f1 / g).powi(2));

        vec![f1, f2]
    })?;

    engine.run()?;

    let front = engine.pareto_front()?;
    println!("pareto front size: {}", front.len());
    for solution in front.iter().take(5) {
        println!(
            "objectives={:?}, rank={}, crowding_distance={:.3}",
            solution.objectives, solution.rank, solution.crowding_distance
        );
    }

    engine.stats.render_report("output/nsga2")?;
    Ok(())
}
```

### 示例 4：单种群（不启用岛模型）

```rust
use genetic_algorithm_rust::{
    CrossoverType, EngineConfig, EvolutionEngine, GeneScalarType, GeneValue, MutationType,
    SelectionType, StopCondition,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 不调用 island_model(...)，默认 num_islands = 1
    let config = EngineConfig::builder(120, 8, 200, 20)
        .init_range(-5.0, 5.0)
        .genes_value_type(genetic_algorithm_rust::GenesValueType::All(GeneScalarType::F64))
        .selection_type(SelectionType::Tournament { k: 3 })
        .crossover(CrossoverType::SinglePoint, 0.85)
        .mutation(
            MutationType::RandomPerturbation {
                min_delta: -0.3,
                max_delta: 0.3,
            },
            0.12,
        )
        .elitism_count(2)
        .random_seed(Some(7))
        .stop_condition(StopCondition::Any {
            target_fitness: None,
            no_improvement_generations: Some(60),
        })
        .build()?;

    let mut ga = EvolutionEngine::new(config, |genes| {
        // Sphere 函数最小化: f(x)=Σx_i^2, 通过取负号转为最大化 fitness
        -genes
            .iter()
            .map(GeneValue::to_f64)
            .map(|x| x * x)
            .sum::<f64>()
    })?;

    ga.run()?;

    let (_, best) = ga.best_solution()?;
    println!("best fitness: {:.6}", best.fitness_or_panic());
    println!(
        "best genes: {:?}",
        best.genes.iter().map(GeneValue::to_f64).collect::<Vec<_>>()
    );

    ga.stats.render_report("output/single-population")?;
    Ok(())
}
```

运行：

```bash
cargo run
```

单目标运行的输出目录通常包含：

- `fitness_history.svg`
- `best_genes_final.svg`
- `best_genes_trajectory.svg`
- `summary.md`

NSGA-II 运行的输出目录通常包含：

- `front_size_history.svg`
- `pareto_front.svg`
- `summary.md`

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
- 优化模式：`optimization_mode(...)`

多目标模式补充说明：

- 使用 `OptimizationMode::Nsga2 { num_objectives }` 启用 NSGA-II
- evaluator 必须返回与 `num_objectives` 一致长度的目标向量
- 目标默认按“越小越好”处理
- NSGA-II 模式下不支持 `StopCondition::TargetFitness(...)`

参数含义：

- `num_islands`: 岛数量（>1 时启用）
- `migration_count`: 每次迁移个体数
- `migration_interval`: 迁移间隔（代）
- `migration_topology`: 迁移拓扑（如 Ring）

## 架构设计

- `EngineConfig`：参数模型与校验入口
- `EngineKernel`：单种群演化执行器
- `IslandEngine`：多岛执行器（岛间迁移）
- `EvolutionEngine`：统一入口（自动分派单岛/多岛/NSGA-II）
- `Evaluation`：统一单目标 / 多目标评估载体
- `ParetoSolution`：对外暴露的非支配解快照
- `RunStats`：按代统计、摘要与报告导出

执行流：

1. 构建并校验配置
2. 初始化种群
3. 并行评估 fitness 或 objectives
4. 选择 / 非支配排序 -> 交叉 -> 变异 -> 精英保留
5. 更新统计并判断停止条件
6. 输出最优解与实验报告

## 未来规划（可选）

- 更丰富的迁移拓扑与异构岛策略
- 更细粒度的并行策略与性能调优开关
- 更完整的 benchmark 与案例集
- 更系统的多目标优化示例与实验模板

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
