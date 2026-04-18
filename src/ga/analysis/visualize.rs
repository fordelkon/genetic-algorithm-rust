use std::{
    fs,
    path::{Path, PathBuf},
};

use plotters::prelude::*;

use crate::ga::analysis::{
    report::{ExperimentSummary, ParetoExperimentSummary},
    stats::RunStats,
};
use crate::ga::error::GaError;
/// SVG report rendering for GA run statistics.

const BACKGROUND: RGBColor = RGBColor(249, 250, 251);
const PANEL: RGBColor = RGBColor(255, 255, 255);
const TEXT: RGBColor = RGBColor(31, 41, 55);
const AXIS: RGBColor = RGBColor(75, 85, 99);
const GRID_BOLD: RGBColor = RGBColor(209, 213, 219);
const GRID_LIGHT: RGBColor = RGBColor(229, 231, 235);
const BEST_LINE: RGBColor = RGBColor(185, 28, 28);
const AVERAGE_LINE: RGBColor = RGBColor(37, 99, 235);
const SMOOTHED_LINE: RGBColor = RGBColor(5, 150, 105);
const BAND_FILL: RGBColor = RGBColor(148, 163, 184);
const BAR_FILL: RGBColor = RGBColor(59, 130, 246);
const ZERO_LINE: RGBColor = RGBColor(107, 114, 128);
const FRONT_COUNT_LINE: RGBColor = RGBColor(124, 58, 237);
const GENE_ELLIPSIS_VISIBLE_PER_SIDE: usize = 4;

/// Rendering parameters for generated report charts.
#[derive(Debug, Clone)]
pub struct VisualizationOptions {
    pub width: u32,
    pub height: u32,
    pub smoothing_window: usize,
    pub trajectory_columns: usize,
    pub trajectory_rows: usize,
}

impl Default for VisualizationOptions {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            smoothing_window: 5,
            trajectory_columns: 3,
            trajectory_rows: 4,
        }
    }
}

pub fn render_report(
    stats: &RunStats,
    output_dir: &Path,
    options: &VisualizationOptions,
) -> Result<(), GaError> {
    fs::create_dir_all(output_dir).map_err(io_error)?;

    if stats.multi_objective.is_some() {
        render_multi_objective_report(stats, output_dir, options)
    } else {
        render_single_objective_report(stats, output_dir, options)
    }
}

fn render_single_objective_report(
    stats: &RunStats,
    output_dir: &Path,
    options: &VisualizationOptions,
) -> Result<(), GaError> {
    if stats.best_fitness_per_generation.is_empty() {
        return Err(GaError::Visualization(
            "cannot render report for an empty run history".into(),
        ));
    }

    render_fitness_history(
        &output_dir.join("fitness_history.svg"),
        stats,
        options.width,
        options.height,
        options.smoothing_window,
    )?;

    let summary = stats.summary()?;
    render_best_genes_final(
        &output_dir.join("best_genes_final.svg"),
        &summary,
        options.width,
        options.height,
    )?;

    render_best_genes_trajectory(
        &output_dir.join("best_genes_trajectory.svg"),
        stats,
        options.width,
        options.height,
        options.trajectory_columns * options.trajectory_rows,
    )?;

    write_single_summary(stats, &output_dir.join("summary.md"))
}

fn render_multi_objective_report(
    stats: &RunStats,
    output_dir: &Path,
    options: &VisualizationOptions,
) -> Result<(), GaError> {
    let multi = stats
        .multi_objective
        .as_ref()
        .ok_or_else(|| GaError::Visualization("multi-objective history is unavailable".into()))?;

    if multi.front_0_size_per_generation.is_empty() {
        return Err(GaError::Visualization(
            "cannot render report for an empty NSGA-II run history".into(),
        ));
    }

    render_front_size_history(
        &output_dir.join("front_size_history.svg"),
        &multi.front_0_size_per_generation,
        &multi.front_count_per_generation,
        options.width,
        options.height,
    )?;

    render_pareto_front(
        &output_dir.join("pareto_front.svg"),
        &multi.final_pareto_front,
        options.width,
        options.height,
    )?;
    render_pareto_priority(
        &output_dir.join("pareto_priority.svg"),
        &multi.final_pareto_front,
        options.width,
        options.height,
    )?;

    write_multi_summary(stats, &output_dir.join("summary.md"))
}

fn render_fitness_history(
    path: &Path,
    stats: &RunStats,
    width: u32,
    height: u32,
    smoothing_window: usize,
) -> Result<(), GaError> {
    let root = SVGBackend::new(path, (width, height)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plotters_error)?;

    let generations = stats.best_fitness_per_generation.len();
    let smoothed_avg = moving_average(&stats.avg_fitness_per_generation, smoothing_window.max(1));

    let mut y_values = Vec::with_capacity(generations * 5);
    for index in 0..generations {
        let avg = stats.avg_fitness_per_generation[index];
        let std_dev = stats.std_fitness_per_generation[index];
        y_values.push(stats.best_fitness_per_generation[index]);
        y_values.push(avg);
        y_values.push(smoothed_avg[index]);
        y_values.push(avg - std_dev);
        y_values.push(avg + std_dev);
    }
    let (y_min, y_max) = padded_range_from_values(&y_values);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "GA Fitness History",
            ("sans-serif", 32).into_font().color(&TEXT),
        )
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(64)
        .build_cartesian_2d(0usize..generations.saturating_sub(1).max(1), y_min..y_max)
        .map_err(plotters_error)?;

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Fitness")
        .label_style(("sans-serif", 18).into_font().color(&TEXT))
        .axis_desc_style(("sans-serif", 20).into_font().color(&TEXT))
        .bold_line_style(GRID_BOLD.mix(0.5))
        .light_line_style(GRID_LIGHT.mix(0.6))
        .axis_style(AXIS.stroke_width(2))
        .y_label_formatter(&|value| format!("{value:.2}"))
        .draw()
        .map_err(plotters_error)?;

    let band_style = BAND_FILL.mix(0.16).filled();
    chart
        .draw_series((0..generations).map(|index| {
            let avg = stats.avg_fitness_per_generation[index];
            let std_dev = stats.std_fitness_per_generation[index];
            Rectangle::new(
                [(index, avg - std_dev), (index + 1, avg + std_dev)],
                band_style,
            )
        }))
        .map_err(plotters_error)?;

    chart
        .draw_series(LineSeries::new(
            stats
                .best_fitness_per_generation
                .iter()
                .enumerate()
                .map(|(index, value)| (index, *value)),
            BEST_LINE.stroke_width(3),
        ))
        .map_err(plotters_error)?
        .label("Best Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], BEST_LINE.stroke_width(3)));

    chart
        .draw_series(LineSeries::new(
            stats
                .avg_fitness_per_generation
                .iter()
                .enumerate()
                .map(|(index, value)| (index, *value)),
            AVERAGE_LINE.stroke_width(2),
        ))
        .map_err(plotters_error)?
        .label("Average Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], AVERAGE_LINE.stroke_width(2)));

    chart
        .draw_series(LineSeries::new(
            smoothed_avg
                .iter()
                .enumerate()
                .map(|(index, value)| (index, *value)),
            SMOOTHED_LINE.stroke_width(2),
        ))
        .map_err(plotters_error)?
        .label("Smoothed Average")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 24, y)], SMOOTHED_LINE.stroke_width(2))
        });

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", 18).into_font().color(&TEXT))
        .background_style(PANEL.mix(0.95))
        .border_style(GRID_BOLD)
        .draw()
        .map_err(plotters_error)?;

    root.present().map_err(plotters_error)
}

fn render_best_genes_final(
    path: &Path,
    summary: &ExperimentSummary,
    width: u32,
    height: u32,
) -> Result<(), GaError> {
    let root = SVGBackend::new(path, (width, height)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plotters_error)?;

    let window = visible_gene_window(summary.best_genes.len(), GENE_ELLIPSIS_VISIBLE_PER_SIDE);
    let visible_values = window
        .slots
        .iter()
        .filter_map(|slot| match slot {
            GeneSlot::Visible(index) => Some(summary.best_genes[*index]),
            GeneSlot::Ellipsis => None,
        })
        .collect::<Vec<_>>();
    let (y_min, y_max) = padded_range_from_values(&visible_values);
    let x_max = window.slots.len().max(1);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Best Solution Genes",
            ("sans-serif", 32).into_font().color(&TEXT),
        )
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(64)
        .build_cartesian_2d(0usize..x_max, y_min.min(0.0)..y_max.max(0.0))
        .map_err(plotters_error)?;

    chart
        .configure_mesh()
        .x_desc("Gene Index")
        .y_desc("Value")
        .x_labels(window.slots.len())
        .x_label_formatter(&|value| match window.slots.get(*value) {
            Some(GeneSlot::Visible(index)) => format!("g{index}"),
            Some(GeneSlot::Ellipsis) => "...".into(),
            None => String::new(),
        })
        .y_label_formatter(&|value| format!("{value:.2}"))
        .label_style(("sans-serif", 18).into_font().color(&TEXT))
        .axis_desc_style(("sans-serif", 20).into_font().color(&TEXT))
        .bold_line_style(GRID_BOLD.mix(0.5))
        .light_line_style(GRID_LIGHT.mix(0.6))
        .axis_style(AXIS.stroke_width(2))
        .draw()
        .map_err(plotters_error)?;

    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(0usize, 0.0), (x_max, 0.0)],
            ZERO_LINE.stroke_width(2),
        )))
        .map_err(plotters_error)?;

    chart
        .draw_series(
            window
                .slots
                .iter()
                .enumerate()
                .filter_map(|(slot_index, slot)| {
                    let GeneSlot::Visible(gene_index) = slot else {
                        return None;
                    };
                    let left = slot_index;
                    let right = slot_index + 1;
                    Some(Rectangle::new(
                        [(left, 0.0), (right, summary.best_genes[*gene_index])],
                        BAR_FILL.mix(0.75).filled(),
                    ))
                }),
        )
        .map_err(plotters_error)?;

    root.present().map_err(plotters_error)
}

fn render_best_genes_trajectory(
    path: &Path,
    stats: &RunStats,
    width: u32,
    height: u32,
    _max_visible_genes: usize,
) -> Result<(), GaError> {
    let root = SVGBackend::new(path, (width, height)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plotters_error)?;

    let generations = stats.best_genes_per_generation.len();
    let total_genes = stats
        .best_genes_per_generation
        .first()
        .map(|genes| genes.len())
        .unwrap_or(0);
    let window = visible_gene_window(total_genes, GENE_ELLIPSIS_VISIBLE_PER_SIDE);
    let visible_indices = window.visible_indices();

    let gene_values = stats
        .best_genes_per_generation
        .iter()
        .flat_map(|genes| {
            visible_indices
                .iter()
                .filter_map(move |index| genes.get(*index))
                .map(|gene| gene.to_f64())
        })
        .collect::<Vec<_>>();
    let (y_min, y_max) = padded_range_from_values(&gene_values);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Best Genes Trajectory",
            ("sans-serif", 32).into_font().color(&TEXT),
        )
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(64)
        .build_cartesian_2d(0usize..generations.saturating_sub(1).max(1), y_min..y_max)
        .map_err(plotters_error)?;

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Gene Value")
        .label_style(("sans-serif", 18).into_font().color(&TEXT))
        .axis_desc_style(("sans-serif", 20).into_font().color(&TEXT))
        .bold_line_style(GRID_BOLD.mix(0.5))
        .light_line_style(GRID_LIGHT.mix(0.6))
        .axis_style(AXIS.stroke_width(2))
        .y_label_formatter(&|value| format!("{value:.2}"))
        .draw()
        .map_err(plotters_error)?;

    for (visible_index, gene_index) in visible_indices.iter().enumerate() {
        let color = trajectory_color(visible_index).stroke_width(2);
        chart
            .draw_series(LineSeries::new(
                stats
                    .best_genes_per_generation
                    .iter()
                    .enumerate()
                    .map(|(generation, genes)| (generation, genes[*gene_index].to_f64())),
                color,
            ))
            .map_err(plotters_error)?;
    }

    if window.omitted_count > 0 {
        root.draw(&Text::new(
            format!("Omitted Genes: {}", window.omitted_count),
            (width as i32 - 280, 40),
            ("sans-serif", 24).into_font().color(&TEXT),
        ))
        .map_err(plotters_error)?;
    } else {
        root.draw(&Text::new(
            "Omitted Genes: 0",
            (width as i32 - 210, 40),
            ("sans-serif", 24).into_font().color(&TEXT),
        ))
        .map_err(plotters_error)?;
    }

    root.present().map_err(plotters_error)
}

fn render_front_size_history(
    path: &Path,
    front_sizes: &[usize],
    front_counts: &[usize],
    width: u32,
    height: u32,
) -> Result<(), GaError> {
    let root = SVGBackend::new(path, (width, height)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plotters_error)?;

    let y_values = front_sizes
        .iter()
        .chain(front_counts.iter())
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let (y_min, y_max) = padded_range_from_values(&y_values);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Pareto Front Size History",
            ("sans-serif", 32).into_font().color(&TEXT),
        )
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(64)
        .build_cartesian_2d(
            0usize..front_sizes.len().saturating_sub(1).max(1),
            y_min..y_max,
        )
        .map_err(plotters_error)?;

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Front 0 Size")
        .label_style(("sans-serif", 18).into_font().color(&TEXT))
        .axis_desc_style(("sans-serif", 20).into_font().color(&TEXT))
        .bold_line_style(GRID_BOLD.mix(0.5))
        .light_line_style(GRID_LIGHT.mix(0.6))
        .axis_style(AXIS.stroke_width(2))
        .y_label_formatter(&|value| format!("{value:.0}"))
        .draw()
        .map_err(plotters_error)?;

    chart
        .draw_series(LineSeries::new(
            front_sizes
                .iter()
                .enumerate()
                .map(|(index, value)| (index, *value as f64)),
            AVERAGE_LINE.stroke_width(3),
        ))
        .map_err(plotters_error)?
        .label("Front 0 Size")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], AVERAGE_LINE.stroke_width(3)));

    chart
        .draw_series(LineSeries::new(
            front_counts
                .iter()
                .enumerate()
                .map(|(index, value)| (index, *value as f64)),
            FRONT_COUNT_LINE.stroke_width(3),
        ))
        .map_err(plotters_error)?
        .label("Front Count")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 24, y)], FRONT_COUNT_LINE.stroke_width(3))
        });

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .label_font(("sans-serif", 18).into_font().color(&TEXT))
        .background_style(PANEL.mix(0.95))
        .border_style(GRID_BOLD)
        .draw()
        .map_err(plotters_error)?;

    root.present().map_err(plotters_error)
}

fn render_pareto_front(
    path: &Path,
    solutions: &[crate::ga::core::pareto::ParetoSolution],
    width: u32,
    height: u32,
) -> Result<(), GaError> {
    let root = SVGBackend::new(path, (width, height)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plotters_error)?;

    if solutions.is_empty() {
        return Err(GaError::Visualization(
            "cannot render an empty Pareto front".into(),
        ));
    }

    let x_values = solutions
        .iter()
        .map(|solution| solution.objectives.first().copied().unwrap_or_default())
        .collect::<Vec<_>>();
    let y_values = solutions
        .iter()
        .map(|solution| solution.objectives.get(1).copied().unwrap_or_default())
        .collect::<Vec<_>>();
    let (x_min, x_max) = padded_range_from_values(&x_values);
    let (y_min, y_max) = padded_range_from_values(&y_values);

    let mut chart = ChartBuilder::on(&root)
        .caption("Pareto Front", ("sans-serif", 32).into_font().color(&TEXT))
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(64)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(plotters_error)?;

    chart
        .configure_mesh()
        .x_desc("Objective 1")
        .y_desc("Objective 2")
        .label_style(("sans-serif", 18).into_font().color(&TEXT))
        .axis_desc_style(("sans-serif", 20).into_font().color(&TEXT))
        .bold_line_style(GRID_BOLD.mix(0.5))
        .light_line_style(GRID_LIGHT.mix(0.6))
        .axis_style(AXIS.stroke_width(2))
        .x_label_formatter(&|value| format!("{value:.2}"))
        .y_label_formatter(&|value| format!("{value:.2}"))
        .draw()
        .map_err(plotters_error)?;

    let crowding_scale = crowding_scale(solutions);
    chart
        .draw_series(solutions.iter().map(|solution| {
            let x = solution.objectives.first().copied().unwrap_or_default();
            let y = solution.objectives.get(1).copied().unwrap_or_default();
            Circle::new(
                (x, y),
                point_radius(solution.crowding_distance, crowding_scale),
                rank_color(solution.rank).mix(0.78).filled(),
            )
        }))
        .map_err(plotters_error)?;

    root.draw(&Text::new(
        "Color = Rank | Size = Crowding Distance",
        (36, 32),
        ("sans-serif", 22).into_font().color(&TEXT),
    ))
    .map_err(plotters_error)?;

    for rank in unique_ranks(solutions).into_iter().take(4).enumerate() {
        let (offset, value) = rank;
        root.draw(&Text::new(
            format!("Rank {value}"),
            (width as i32 - 180, 44 + offset as i32 * 24),
            ("sans-serif", 18).into_font().color(&rank_color(value)),
        ))
        .map_err(plotters_error)?;
    }

    root.present().map_err(plotters_error)
}

fn render_pareto_priority(
    path: &Path,
    solutions: &[crate::ga::core::pareto::ParetoSolution],
    width: u32,
    height: u32,
) -> Result<(), GaError> {
    let root = SVGBackend::new(path, (width, height)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plotters_error)?;

    if solutions.is_empty() {
        return Err(GaError::Visualization(
            "cannot render NSGA-II priority for an empty Pareto front".into(),
        ));
    }

    let mut ranked = solutions.iter().collect::<Vec<_>>();
    ranked.sort_by(compare_nsga2_priority);

    let crowding_values = ranked
        .iter()
        .map(|solution| crowding_for_chart(solution.crowding_distance))
        .collect::<Vec<_>>();
    let (_, y_max) = padded_range_from_values(&crowding_values);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "NSGA-II Priority",
            ("sans-serif", 32).into_font().color(&TEXT),
        )
        .margin(24)
        .x_label_area_size(48)
        .y_label_area_size(72)
        .build_cartesian_2d(0usize..ranked.len().max(1), 0.0..y_max.max(1.0))
        .map_err(plotters_error)?;

    chart
        .configure_mesh()
        .x_desc("Priority Order")
        .y_desc("Crowding Distance")
        .x_label_formatter(&|value| format!("#{value}"))
        .y_label_formatter(&|value| format!("{value:.2}"))
        .label_style(("sans-serif", 18).into_font().color(&TEXT))
        .axis_desc_style(("sans-serif", 20).into_font().color(&TEXT))
        .bold_line_style(GRID_BOLD.mix(0.5))
        .light_line_style(GRID_LIGHT.mix(0.6))
        .axis_style(AXIS.stroke_width(2))
        .draw()
        .map_err(plotters_error)?;

    chart
        .draw_series(ranked.iter().enumerate().map(|(index, solution)| {
            Rectangle::new(
                [
                    (index, 0.0),
                    (index + 1, crowding_for_chart(solution.crowding_distance)),
                ],
                rank_color(solution.rank).mix(0.82).filled(),
            )
        }))
        .map_err(plotters_error)?;

    root.draw(&Text::new(
        "Sorted by Rank asc, then Crowding Distance desc",
        (36, 32),
        ("sans-serif", 22).into_font().color(&TEXT),
    ))
    .map_err(plotters_error)?;
    root.draw(&Text::new(
        "Higher bars are preferred within the same rank",
        (36, 58),
        ("sans-serif", 18).into_font().color(&AXIS),
    ))
    .map_err(plotters_error)?;

    root.present().map_err(plotters_error)
}

fn moving_average(values: &[f64], window: usize) -> Vec<f64> {
    values
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let start = index.saturating_sub(window.saturating_sub(1));
            let slice = &values[start..=index];
            slice.iter().sum::<f64>() / slice.len() as f64
        })
        .collect()
}

fn trajectory_color(index: usize) -> RGBColor {
    const COLORS: [RGBColor; 8] = [
        RGBColor(37, 99, 235),
        RGBColor(5, 150, 105),
        RGBColor(220, 38, 38),
        RGBColor(217, 119, 6),
        RGBColor(124, 58, 237),
        RGBColor(8, 145, 178),
        RGBColor(225, 29, 72),
        RGBColor(22, 163, 74),
    ];
    COLORS[index % COLORS.len()]
}

fn rank_color(rank: usize) -> RGBColor {
    const COLORS: [RGBColor; 6] = [
        RGBColor(220, 38, 38),
        RGBColor(37, 99, 235),
        RGBColor(5, 150, 105),
        RGBColor(217, 119, 6),
        RGBColor(124, 58, 237),
        RGBColor(8, 145, 178),
    ];
    COLORS[rank % COLORS.len()]
}

fn crowding_scale(solutions: &[crate::ga::core::pareto::ParetoSolution]) -> f64 {
    let max_finite = solutions
        .iter()
        .map(|solution| solution.crowding_distance)
        .filter(|value| value.is_finite())
        .fold(0.0, f64::max);

    if max_finite > 0.0 { max_finite } else { 1.0 }
}

fn crowding_for_chart(crowding_distance: f64) -> f64 {
    if crowding_distance.is_finite() {
        crowding_distance.max(0.0)
    } else {
        1.0
    }
}

fn point_radius(crowding_distance: f64, crowding_scale: f64) -> i32 {
    let normalized = if crowding_distance.is_finite() {
        (crowding_distance / crowding_scale).clamp(0.0, 1.0)
    } else {
        1.0
    };

    4 + (normalized * 8.0).round() as i32
}

fn compare_nsga2_priority(
    left: &&crate::ga::core::pareto::ParetoSolution,
    right: &&crate::ga::core::pareto::ParetoSolution,
) -> std::cmp::Ordering {
    left.rank
        .cmp(&right.rank)
        .then_with(|| right.crowding_distance.total_cmp(&left.crowding_distance))
}

fn unique_ranks(solutions: &[crate::ga::core::pareto::ParetoSolution]) -> Vec<usize> {
    let mut ranks = solutions
        .iter()
        .map(|solution| solution.rank)
        .collect::<Vec<_>>();
    ranks.sort_unstable();
    ranks.dedup();
    ranks
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GeneSlot {
    Visible(usize),
    Ellipsis,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VisibleGeneWindow {
    slots: Vec<GeneSlot>,
    omitted_count: usize,
}

impl VisibleGeneWindow {
    fn visible_indices(&self) -> Vec<usize> {
        self.slots
            .iter()
            .filter_map(|slot| match slot {
                GeneSlot::Visible(index) => Some(*index),
                GeneSlot::Ellipsis => None,
            })
            .collect()
    }
}

fn visible_gene_window(total_genes: usize, keep_each_side: usize) -> VisibleGeneWindow {
    if total_genes <= keep_each_side * 2 {
        return VisibleGeneWindow {
            slots: (0..total_genes).map(GeneSlot::Visible).collect(),
            omitted_count: 0,
        };
    }

    let mut slots = (0..keep_each_side)
        .map(GeneSlot::Visible)
        .collect::<Vec<_>>();
    slots.push(GeneSlot::Ellipsis);
    slots.extend((total_genes - keep_each_side..total_genes).map(GeneSlot::Visible));

    VisibleGeneWindow {
        slots,
        omitted_count: total_genes.saturating_sub(keep_each_side * 2),
    }
}

fn padded_range_from_values(values: &[f64]) -> (f64, f64) {
    let mut min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let mut max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !min.is_finite() || !max.is_finite() {
        return (-1.0, 1.0);
    }

    if (max - min).abs() < f64::EPSILON {
        let pad = if min.abs() < 1.0 {
            1.0
        } else {
            min.abs() * 0.1
        };
        min -= pad;
        max += pad;
        return (min, max);
    }

    let pad = (max - min) * 0.1;
    (min - pad, max + pad)
}

fn write_single_summary(stats: &RunStats, path: &Path) -> Result<(), GaError> {
    let summary = ExperimentSummary::from_stats(stats)?;
    fs::write(
        path,
        format!(
            "# GA Summary\n\ngenerations: {}\nbest_fitness: {:.6}\nfinal_avg_fitness: {:.6}\nfinal_std_fitness: {:.6}\n",
            summary.generations,
            summary.best_fitness,
            summary.final_avg_fitness,
            summary.final_std_fitness
        ),
    )
    .map_err(io_error)
}

fn write_multi_summary(stats: &RunStats, path: &Path) -> Result<(), GaError> {
    let summary = ParetoExperimentSummary::from_stats(stats)?;
    fs::write(
        path,
        format!(
            "# NSGA-II Summary\n\ngenerations: {}\nfinal_front_size: {}\nfinal_front_count: {}\n",
            summary.generations, summary.final_front_size, summary.final_front_count
        ),
    )
    .map_err(io_error)
}

fn io_error(error: std::io::Error) -> GaError {
    GaError::Visualization(error.to_string())
}

fn plotters_error<E>(error: DrawingAreaErrorKind<E>) -> GaError
where
    E: std::error::Error + Send + Sync + 'static,
{
    GaError::Visualization(format!("{error:?}"))
}

#[allow(dead_code)]
fn _resolve(path: &Path) -> PathBuf {
    path.to_path_buf()
}
