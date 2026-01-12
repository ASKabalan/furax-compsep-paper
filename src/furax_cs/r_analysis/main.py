import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from furax_cs.data.instruments import get_instrument

from ..logging_utils import (
    error,
    warning,
)
from .compute import get_compute_flags
from .parser import parse_args
from .plotting import (
    get_plot_flags,
    run_plot,
)
from .r_estimate import run_estimate
from .run_grep import run_grep
from .snapshot import (
    run_snapshot,
)
from .validate import run_validate

out_folder = "plots/"


def run_analysis():
    """Entry point for the r_analysis CLI driver.

    This function orchestrates the complete analysis pipeline using subcommands:
    - snap: Compute statistics and save snapshot (auto-caches W_D_FG)
    - plot: Generate plots from results or snapshots
    - validate: Run NLL validation analysis
    - estimate: Estimate tensor-to-scalar ratio r from spectra or maps

    The analysis flow is controlled by command-line arguments parsed via :func:`parse_args`.
    """
    args = parse_args()

    # Handle estimate subcommand separately (doesn't use common_parser)
    if args.subcommand == "estimate":
        return run_estimate(
            cmb_path=args.cmb,
            cmb_hat_path=args.cmb_hat,
            syst_path=args.syst,
            fsky=args.fsky,
            nside=args.nside,
            output_path=args.output,
            output_format=args.output_format,
        )

    # For other subcommands, get common arguments
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
    if args.subcommand == "snap":
        flags = get_compute_flags(args, snapshot_mode=True)  # compute everything
        return run_snapshot(
            matched_results,
            nside,
            instrument,
            args.output_snapshot,
            flags,
            args.max_iterations,
            args.solver,
        )

    if args.subcommand == "plot":
        # Handle titles: if regex expanded to different number of groups, use expanded names
        titles = args.titles
        if not titles or len(titles) != len(matched_results):
            if titles and len(titles) != len(matched_results):
                warning(
                    f"Got {len(matched_results)} result groups but {len(titles)} titles. "
                    f"Using expanded pattern names as titles."
                )
            titles = list(matched_results.keys())

        flags = get_compute_flags(args, snapshot_mode=False)
        indiv_flags, aggregate_flags = get_plot_flags(args)
        return run_plot(
            matched_results,
            titles,
            nside,
            instrument,
            args.snapshot,
            flags,
            indiv_flags,
            aggregate_flags,
            args.max_iterations,
            args.solver,
            args.output_format,
            args.font_size,
        )

    if args.subcommand == "validate":
        # Handle titles: if regex expanded to different number of groups, use expanded names
        titles = args.titles
        if not titles or len(titles) != len(matched_results):
            if titles and len(titles) != len(matched_results):
                warning(
                    f"Got {len(matched_results)} result groups but {len(titles)} titles. "
                    f"Using expanded pattern names as titles."
                )
            titles = list(matched_results.keys())
        return run_validate(
            matched_results,
            titles,
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
