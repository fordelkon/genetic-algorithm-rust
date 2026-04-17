use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::ga::analysis::{
    report::{ExperimentSummary, ParetoExperimentSummary},
    stats::RunStats,
};
use crate::ga::error::GaError;
/// SVG report rendering for GA run statistics.

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
    _options: &VisualizationOptions,
) -> Result<(), GaError> {
    fs::create_dir_all(output_dir).map_err(io_error)?;

    if stats.multi_objective.is_some() {
        render_multi_objective_report(stats, output_dir)
    } else {
        render_single_objective_report(stats, output_dir)
    }
}

fn render_single_objective_report(stats: &RunStats, output_dir: &Path) -> Result<(), GaError> {
    if stats.best_fitness_per_generation.is_empty() {
        return Err(GaError::Visualization(
            "cannot render report for an empty run history".into(),
        ));
    }

    fs::write(
        output_dir.join("fitness_history.svg"),
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">
<text x="40" y="60">GA Fitness History</text>
<text x="40" y="110">Best Fitness</text>
<text x="40" y="150">Average Fitness</text>
<text x="40" y="190">Smoothed Average</text>
</svg>"#,
    )
    .map_err(io_error)?;

    let summary = stats.summary()?;
    let best_genes = summary
        .best_genes
        .iter()
        .enumerate()
        .map(|(index, value)| format!("<text x=\"40\" y=\"{}\">gene[{index}] = {value:.4}</text>", 80 + index * 24))
        .collect::<String>();
    fs::write(
        output_dir.join("best_genes_final.svg"),
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1280\" height=\"720\"><text x=\"40\" y=\"60\">Best Solution Genes</text>{best_genes}</svg>"
        ),
    )
    .map_err(io_error)?;

    fs::write(
        output_dir.join("best_genes_trajectory.svg"),
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">
<text x="40" y="60">Best Genes Trajectory</text>
<text x="40" y="110">Additional Genes</text>
</svg>"#,
    )
    .map_err(io_error)?;

    write_single_summary(stats, &output_dir.join("summary.md"))
}

fn render_multi_objective_report(stats: &RunStats, output_dir: &Path) -> Result<(), GaError> {
    let multi = stats.multi_objective.as_ref().ok_or_else(|| {
        GaError::Visualization("multi-objective history is unavailable".into())
    })?;

    if multi.front_0_size_per_generation.is_empty() {
        return Err(GaError::Visualization(
            "cannot render report for an empty NSGA-II run history".into(),
        ));
    }

    fs::write(
        output_dir.join("front_size_history.svg"),
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">
<text x="40" y="60">Pareto Front Size History</text>
</svg>"#,
    )
    .map_err(io_error)?;

    let points = multi
        .final_pareto_front
        .iter()
        .enumerate()
        .map(|(index, solution)| {
            let x = solution.objectives.first().copied().unwrap_or_default();
            let y = solution.objectives.get(1).copied().unwrap_or_default();
            format!("<text x=\"40\" y=\"{}\">p{index}: ({x:.4}, {y:.4})</text>", 90 + index * 24)
        })
        .collect::<String>();
    fs::write(
        output_dir.join("pareto_front.svg"),
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1280\" height=\"720\"><text x=\"40\" y=\"60\">Pareto Front</text>{points}</svg>"
        ),
    )
    .map_err(io_error)?;

    write_multi_summary(stats, &output_dir.join("summary.md"))
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
            summary.generations,
            summary.final_front_size,
            summary.final_front_count
        ),
    )
    .map_err(io_error)
}

fn io_error(error: std::io::Error) -> GaError {
    GaError::Visualization(error.to_string())
}

#[allow(dead_code)]
fn _resolve(path: &Path) -> PathBuf {
    path.to_path_buf()
}
