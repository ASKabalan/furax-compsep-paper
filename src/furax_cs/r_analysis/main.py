import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from furax_cs.data.instruments import get_instrument

from .caching import cache_systematics
from .compute import get_compute_flags
from .logging_utils import (
    error,
)
from .parser import parse_args
from .plotting import (
    get_plot_flags,
    run_plot,
)
from .run_grep import run_grep
from .snapshot import (
    run_snapshot,
)
from .validate import run_validate

out_folder = "plots/"


def run_analysis():
    """Entry point for the r_analysis CLI driver.

    This function orchestrates the complete analysis pipeline using subcommands:
    - cache: Compute and cache W_D_FG systematics
    - snap: Compute statistics and save snapshot
    - plot: Generate plots from results or snapshots
    - validate: Run NLL validation analysis

    The analysis flow is controlled by command-line arguments parsed via :func:`parse_args`.
    """
    args = parse_args()
    nside = args.nside
    instrument = get_instrument(args.instrument)
    if args.no_tex:
        plt.rcParams["text.usetex"] = False

    # Grep results using the new consolidated function
    matched_results = run_grep(
        result_folders=args.input_results_dir,
        run_specs=args.runs,
    )
    if len(matched_results) == 0:
        error("No results matched the provided run specifications. Exiting.")
        return

    # Dispatch to subcommand handler
    if args.subcommand == "cache":
        return cache_systematics(
            matched_results,
            nside,
            instrument,
            force_recompute=args.force,
            max_iter=args.max_iterations,
        )

    if args.subcommand == "snap":
        flags = get_compute_flags(args, snapshot_mode=True)  # compute everything
        return run_snapshot(
            matched_results,
            nside,
            instrument,
            args.output_snapshot,
            flags,
            args.max_iterations,
        )

    if args.subcommand == "plot":
        flags = get_compute_flags(args, snapshot_mode=False)
        indiv_flags, aggregate_flags = get_plot_flags(args)
        return run_plot(
            matched_results,
            args.titles,
            nside,
            instrument,
            args.snapshot,
            flags,
            indiv_flags,
            aggregate_flags,
            args.max_iterations,
            args.output_format,
            args.font_size,
        )

    if args.subcommand == "validate":
        return run_validate(
            matched_results,
            args.titles,
            nside,
            instrument,
            args.steps,
            args.noise_ratio,
            args.scales,
        )


def main():
    """CLI entry point for the r_analysis tool.

    This is the primary entry point registered as the ``r_analysis`` console script
    in pyproject.toml. It simply invokes :func:`run_analysis`.
    """
    run_analysis()


if __name__ == "__main__":
    run_analysis()
