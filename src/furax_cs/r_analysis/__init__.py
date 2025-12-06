import os

os.environ["EQX_ON_ERROR"] = "nan"

from .compute import compute_all, compute_group, compute_single_folder, get_compute_flags
from .main import run_analysis
from .parser import parse_args
from .plotting import (
    plot_all_cl_residuals,
    plot_all_cmb,
    plot_all_r_estimation,
    plot_r_vs_clusters,
    plot_variance_vs_clusters,
    plot_variance_vs_r,
    run_plot,
)
from .r_estimate import estimate_r, get_camb_templates, log_likelihood
from .residuals import (
    compute_cl_bb_sum,
    compute_cl_obs_bb,
    compute_cl_true_bb,
    compute_statistical_res,
    compute_systematic_res,
    compute_total_res,
)
from .snapshot import (
    load_and_filter_snapshot,
    load_snapshot,
    run_snapshot,
    save_snapshot,
    save_snapshot_entry,
    write_snapshot_manifest,
)
from .utils import expand_stokes, params_to_maps

__all__ = [
    "run_analysis",
    "compute_results",
    "compute_all",
    "compute_group",
    "compute_single_folder",
    "get_compute_flags",
    "plot_results",
    "parse_args",
    "estimate_r",
    "get_camb_templates",
    "log_likelihood",
    "compute_systematic_res",
    "compute_statistical_res",
    "compute_total_res",
    "compute_cl_bb_sum",
    "compute_cl_obs_bb",
    "compute_cl_true_bb",
    "compute_w",
    "cache_expensive_computations",
    "run_cache",
    "load_snapshot",
    "load_and_filter_snapshot",
    "run_snapshot",
    "save_snapshot",
    "save_snapshot_entry",
    "write_snapshot_manifest",
    "plot_all_cl_residuals",
    "plot_all_cmb",
    "plot_all_r_estimation",
    "plot_r_vs_clusters",
    "plot_variance_vs_clusters",
    "plot_variance_vs_r",
    "run_plot",
    "parse_run_spec",
    "parse_filter_kw",
    "matches_filter",
    "expand_run_specs",
    "grep_results",
    "expand_stokes",
    "params_to_maps",
    "run_validate",
    "run_grep",
]
