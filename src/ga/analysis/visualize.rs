use std::{
    fs,
    path::{Path, PathBuf},
};

use plotters::{
    prelude::*,
    series::{AreaSeries, LineSeries},
};

use crate::ga::analysis::{report::ExperimentSummary, stats::RunStats};
use crate::ga::error::GaError;
/// SVG report rendering for GA run statistics.

/// Rendering parameters for generated report charts.
#[derive(Debug, Clone)]
pub struct VisualizationOptions {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Moving-average window size for smoothing.
    pub smoothing_window: usize,
    /// Number of columns in gene-trajectory panels.
    pub trajectory_columns: usize,
    /// Number of rows in gene-trajectory panels.
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

const BACKGROUND: RGBColor = RGBColor(250, 250, 247);
const TEXT_COLOR: RGBColor = RGBColor(33, 37, 41);
const GRID_COLOR: RGBColor = RGBColor(217, 217, 217);
const BEST_COLOR: RGBColor = RGBColor(31, 119, 180);
const AVG_COLOR: RGBColor = RGBColor(127, 140, 141);
const BAR_POSITIVE: RGBColor = RGBColor(31, 119, 180);
const BAR_NEGATIVE: RGBColor = RGBColor(214, 39, 40);
const BAND_COLOR: RGBColor = RGBColor(158, 182, 197);
const TRAJECTORY_COLORS: [RGBColor; 8] = [
    RGBColor(31, 119, 180),
    RGBColor(255, 127, 14),
    RGBColor(44, 160, 44),
    RGBColor(214, 39, 40),
    RGBColor(148, 103, 189),
    RGBColor(140, 86, 75),
    RGBColor(227, 119, 194),
    RGBColor(23, 190, 207),
];

/// Renders the full report bundle into the provided directory.
pub fn render_report(
    stats: &RunStats,
    output_dir: &Path,
    options: &VisualizationOptions,
) -> Result<(), GaError> {
    if stats.best_fitness_per_generation.is_empty() {
        return Err(GaError::Visualization(
            "cannot render report for an empty run history".into(),
        ));
    }

    fs::create_dir_all(output_dir).map_err(io_error)?;

    render_fitness_history(stats, &output_dir.join("fitness_history.svg"), options)?;
    render_best_genes_final(stats, &output_dir.join("best_genes_final.svg"), options)?;
    render_best_genes_trajectory(
        stats,
        &output_dir.join("best_genes_trajectory.svg"),
        options,
    )?;
    write_summary(stats, &output_dir.join("summary.md"))?;

    Ok(())
}

fn render_fitness_history(
    stats: &RunStats,
    path: &Path,
    options: &VisualizationOptions,
) -> Result<(), GaError> {
    let root = SVGBackend::new(path, (options.width, options.height)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_error)?;

    let generations = stats.best_fitness_per_generation.len();
    let x_end = (generations.saturating_sub(1)).max(1) as i32;
    let mut all_values = stats.best_fitness_per_generation.clone();
    all_values.extend(stats.avg_fitness_per_generation.iter().copied());
    all_values.extend(
        stats
            .avg_fitness_per_generation
            .iter()
            .zip(stats.std_fitness_per_generation.iter())
            .flat_map(|(avg, std_dev)| [avg - std_dev, avg + std_dev]),
    );
    let (y_min, y_max) = padded_range(&all_values);
    let smoothed_avg = moving_average(&stats.avg_fitness_per_generation, options.smoothing_window);

    let mut chart = ChartBuilder::on(&root)
        .margin(30)
        .x_label_area_size(42)
        .y_label_area_size(72)
        .caption(
            "GA Fitness History",
            ("sans-serif", 28).into_font().color(&TEXT_COLOR),
        )
        .build_cartesian_2d(0..x_end, y_min..y_max)
        .map_err(plot_error)?;

    chart
        .configure_mesh()
        .x_desc("Generation")
        .y_desc("Fitness")
        .axis_style(TEXT_COLOR.stroke_width(1))
        .light_line_style(GRID_COLOR.mix(0.35))
        .bold_line_style(GRID_COLOR.mix(0.65))
        .label_style(("sans-serif", 16).into_font().color(&TEXT_COLOR))
        .draw()
        .map_err(plot_error)?;

    chart
        .draw_series(AreaSeries::new(
            stats
                .avg_fitness_per_generation
                .iter()
                .zip(stats.std_fitness_per_generation.iter())
                .enumerate()
                .map(|(idx, (avg, std_dev))| (idx as i32, avg + std_dev)),
            0.0,
            BAND_COLOR.mix(0.10),
        ))
        .map_err(plot_error)?;

    chart
        .draw_series(AreaSeries::new(
            stats
                .avg_fitness_per_generation
                .iter()
                .zip(stats.std_fitness_per_generation.iter())
                .enumerate()
                .map(|(idx, (avg, std_dev))| (idx as i32, avg - std_dev)),
            0.0,
            BACKGROUND,
        ))
        .map_err(plot_error)?;

    chart
        .draw_series(LineSeries::new(
            stats
                .best_fitness_per_generation
                .iter()
                .enumerate()
                .map(|(idx, value)| (idx as i32, *value)),
            BEST_COLOR.stroke_width(3),
        ))
        .map_err(plot_error)?
        .label("Best Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], BEST_COLOR.stroke_width(3)));

    chart
        .draw_series(LineSeries::new(
            stats
                .avg_fitness_per_generation
                .iter()
                .enumerate()
                .map(|(idx, value)| (idx as i32, *value)),
            AVG_COLOR.stroke_width(3),
        ))
        .map_err(plot_error)?
        .label("Average Fitness")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], AVG_COLOR.stroke_width(3)));

    chart
        .draw_series(LineSeries::new(
            smoothed_avg
                .iter()
                .enumerate()
                .map(|(idx, value)| (idx as i32, *value)),
            BEST_COLOR.mix(0.55).stroke_width(4),
        ))
        .map_err(plot_error)?
        .label("Smoothed Average")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 24, y)],
                BEST_COLOR.mix(0.55).stroke_width(4),
            )
        });

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(TEXT_COLOR.stroke_width(1))
        .background_style(BACKGROUND.mix(0.9))
        .label_font(("sans-serif", 16).into_font())
        .draw()
        .map_err(plot_error)?;

    root.present().map_err(plot_error)
}

fn render_best_genes_final(
    stats: &RunStats,
    path: &Path,
    options: &VisualizationOptions,
) -> Result<(), GaError> {
    let summary = stats.summary()?;
    let root = SVGBackend::new(path, (options.width, options.height)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_error)?;

    let gene_values = summary.best_genes;
    let x_end = gene_values.len().max(1) as i32;
    let mut y_values = gene_values.clone();
    y_values.push(0.0);
    let (y_min, y_max) = padded_range(&y_values);

    let mut chart = ChartBuilder::on(&root)
        .margin(30)
        .x_label_area_size(42)
        .y_label_area_size(72)
        .caption(
            "Best Solution Genes",
            ("sans-serif", 28).into_font().color(&TEXT_COLOR),
        )
        .build_cartesian_2d(0..x_end, y_min..y_max)
        .map_err(plot_error)?;

    chart
        .configure_mesh()
        .x_desc("Gene Index")
        .y_desc("Gene Value")
        .axis_style(TEXT_COLOR.stroke_width(1))
        .light_line_style(GRID_COLOR.mix(0.35))
        .bold_line_style(GRID_COLOR.mix(0.65))
        .label_style(("sans-serif", 16).into_font().color(&TEXT_COLOR))
        .draw()
        .map_err(plot_error)?;

    chart
        .draw_series(gene_values.iter().enumerate().map(|(idx, value)| {
            let x0 = idx as i32;
            let x1 = x0 + 1;
            let style = if *value >= 0.0 {
                BAR_POSITIVE.filled()
            } else {
                BAR_NEGATIVE.filled()
            };
            Rectangle::new([(x0, 0.0), (x1, *value)], style)
        }))
        .map_err(plot_error)?;

    root.present().map_err(plot_error)
}

fn render_best_genes_trajectory(
    stats: &RunStats,
    path: &Path,
    options: &VisualizationOptions,
) -> Result<(), GaError> {
    let generation_count = stats.best_genes_per_generation.len();
    if generation_count == 0 {
        return Err(GaError::Visualization(
            "missing best gene trajectory history".into(),
        ));
    }

    let gene_count = stats.best_genes_per_generation[0].len();
    let total_slots = options.trajectory_columns * options.trajectory_rows;
    let visible_gene_slots = if gene_count > total_slots {
        total_slots.saturating_sub(1)
    } else {
        total_slots
    };
    let visible_gene_count = gene_count.min(visible_gene_slots);
    let gene_indices = (0..visible_gene_count).collect::<Vec<_>>();
    let omitted_gene_indices = (visible_gene_count..gene_count).collect::<Vec<_>>();
    let mut all_values = Vec::new();
    for generation in &stats.best_genes_per_generation {
        for gene in generation {
            all_values.push(gene.to_f64());
        }
    }
    let (y_min, y_max) = padded_range(&all_values);
    let root = SVGBackend::new(path, (options.width, options.height)).into_drawing_area();
    root.fill(&BACKGROUND).map_err(plot_error)?;
    let titled = root
        .titled(
            "Best Genes Trajectory",
            ("sans-serif", 24).into_font().color(&TEXT_COLOR),
        )
        .map_err(plot_error)?;
    let areas = titled.split_evenly((options.trajectory_rows, options.trajectory_columns));
    let x_end = (generation_count.saturating_sub(1)).max(1) as i32;

    for (slot_idx, area) in areas.into_iter().enumerate() {
        if let Some(&gene_idx) = gene_indices.get(slot_idx) {
            render_gene_panel(stats, area, gene_idx, x_end, (y_min, y_max), options)?;
        } else if slot_idx == visible_gene_count && !omitted_gene_indices.is_empty() {
            render_omission_card(area, &omitted_gene_indices)?;
        } else {
            area.fill(&BACKGROUND).map_err(plot_error)?;
        }
    }

    root.present().map_err(plot_error)
}

fn write_summary(stats: &RunStats, path: &PathBuf) -> Result<(), GaError> {
    let summary = ExperimentSummary::from_stats(stats)?;
    let markdown = format!(
        "# Genetic Algorithm Experiment Summary\n\n\
        - Generations: {}\n\
        - Best fitness: {:.6}\n\
        - Final average fitness: {:.6}\n\
        - Final fitness std-dev: {:.6}\n\
        - Initial best fitness: {:.6}\n\
        - Final best fitness: {:.6}\n\
        - Improvement: {:.6}\n\
        - Best genes: {:?}\n",
        summary.generations,
        summary.best_fitness,
        summary.final_avg_fitness,
        summary.final_std_fitness,
        summary.initial_best_fitness,
        summary.final_best_fitness,
        summary.improvement,
        summary.best_genes
    );

    fs::write(path, markdown).map_err(io_error)
}

fn padded_range(values: &[f64]) -> (f64, f64) {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if !min.is_finite() || !max.is_finite() {
        return (-1.0, 1.0);
    }

    let spread = (max - min).abs();
    let pad = if spread < f64::EPSILON {
        max.abs().max(1.0) * 0.1
    } else {
        spread * 0.1
    };

    (min - pad, max + pad)
}

fn io_error(error: std::io::Error) -> GaError {
    GaError::Visualization(error.to_string())
}

fn plot_error<E: std::fmt::Display>(error: E) -> GaError {
    GaError::Visualization(error.to_string())
}

fn moving_average(values: &[f64], window: usize) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let effective_window = window.max(1);
    let mut smoothed = Vec::with_capacity(values.len());

    for idx in 0..values.len() {
        let start = idx.saturating_sub(effective_window.saturating_sub(1));
        let slice = &values[start..=idx];
        smoothed.push(slice.iter().sum::<f64>() / slice.len() as f64);
    }

    smoothed
}

fn render_gene_panel(
    stats: &RunStats,
    area: DrawingArea<SVGBackend<'_>, plotters::coord::Shift>,
    gene_idx: usize,
    x_end: i32,
    y_range: (f64, f64),
    options: &VisualizationOptions,
) -> Result<(), GaError> {
    let mut chart = ChartBuilder::on(&area)
        .margin(16)
        .x_label_area_size(28)
        .y_label_area_size(42)
        .caption(
            format!("Gene {gene_idx}"),
            ("sans-serif", 17).into_font().color(&TEXT_COLOR),
        )
        .build_cartesian_2d(0..x_end, y_range.0..y_range.1)
        .map_err(plot_error)?;

    chart
        .configure_mesh()
        .x_desc("Gen")
        .y_desc("Value")
        .axis_style(TEXT_COLOR.stroke_width(1))
        .light_line_style(GRID_COLOR.mix(0.25))
        .bold_line_style(GRID_COLOR.mix(0.45))
        .label_style(("sans-serif", 11).into_font().color(&TEXT_COLOR))
        .draw()
        .map_err(plot_error)?;

    let series = stats
        .best_genes_per_generation
        .iter()
        .enumerate()
        .map(|(generation_idx, genes)| (generation_idx as i32, genes[gene_idx].to_f64()))
        .collect::<Vec<_>>();
    let smoothed = moving_average(
        &series.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
        options.smoothing_window,
    );
    let color = TRAJECTORY_COLORS[gene_idx % TRAJECTORY_COLORS.len()];

    chart
        .draw_series(LineSeries::new(
            series.iter().copied(),
            color.mix(0.35).stroke_width(2),
        ))
        .map_err(plot_error)?;
    chart
        .draw_series(LineSeries::new(
            smoothed
                .iter()
                .enumerate()
                .map(|(idx, value)| (idx as i32, *value)),
            color.stroke_width(3),
        ))
        .map_err(plot_error)?;

    Ok(())
}

fn render_omission_card(
    area: DrawingArea<SVGBackend<'_>, plotters::coord::Shift>,
    omitted_gene_indices: &[usize],
) -> Result<(), GaError> {
    area.fill(&BACKGROUND).map_err(plot_error)?;
    let (width, height) = area.dim_in_pixel();
    area.draw(&Rectangle::new(
        [(6, 6), (width as i32 - 6, height as i32 - 6)],
        ShapeStyle::from(&GRID_COLOR).stroke_width(1),
    ))
    .map_err(plot_error)?;

    let preview = omitted_gene_indices
        .iter()
        .take(5)
        .map(|index| index.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let suffix = if omitted_gene_indices.len() > 5 {
        ", ..."
    } else {
        ""
    };

    area.draw(&Text::new(
        "Additional Genes",
        (24, 42),
        ("sans-serif", 18).into_font().color(&TEXT_COLOR),
    ))
    .map_err(plot_error)?;
    area.draw(&Text::new(
        format!("{} genes omitted", omitted_gene_indices.len()),
        (24, 82),
        ("sans-serif", 14).into_font().color(&TEXT_COLOR),
    ))
    .map_err(plot_error)?;
    area.draw(&Text::new(
        format!("Omitted: {preview}{suffix}"),
        (24, 116),
        ("sans-serif", 13).into_font().color(&TEXT_COLOR),
    ))
    .map_err(plot_error)?;
    area.draw(&Text::new(
        "Increase the grid size to inspect all genes.",
        (24, 150),
        ("sans-serif", 13).into_font().color(&AVG_COLOR),
    ))
    .map_err(plot_error)?;

    Ok(())
}
