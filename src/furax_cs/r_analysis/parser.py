import argparse


def parse_args():
    """Parse command-line arguments for the r_analysis tool.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments with fields controlling input data,
        computation options, and plotting behavior.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark FGBuster and Furax Component Separation Methods"
    )

    parser.add_argument("-n", "--nside", type=int, default=64, help="The nside of the map")
    parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        default="LiteBIRD",
        choices=["LiteBIRD", "Planck", "default"],
        help="Instrument to use",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=str,
        nargs="*",
        help="List of run name keywords to filter result folders",
    )
    parser.add_argument(
        "-t",
        "--titles",
        type=str,
        nargs="*",
        help="List of titles for the plots",
    )
    parser.add_argument(
        "-pi",
        "--plot-illustrations",
        action="store_true",
        help="Plot illustrations of the results",
    )
    # Individual plot toggles (single-run style)
    parser.add_argument(
        "-pp",
        "--plot-params",
        action="store_true",
        help="Plot only spectral parameter maps (beta_d, T_d, beta_s)",
    )
    parser.add_argument(
        "-pt",
        "--plot-patches",
        action="store_true",
        help="Plot only patch assignments for each parameter",
    )
    parser.add_argument(
        "-pv",
        "--plot-validation-curves",
        action="store_true",
        help="Plot validation curves of the results",
    )
    parser.add_argument(
        "-ps",
        "--plot-cl-spectra",
        action="store_true",
        help="Plot spectra of the results one by one",
    )
    parser.add_argument(
        "-pc",
        "--plot-cmb-recon",
        action="store_true",
        help="Plot CMB reconstructions of the results one by one",
    )
    parser.add_argument(
        "-psm",
        "--plot-systematic-maps",
        action="store_true",
        help="Plot systematic residual maps (single run)",
    )
    parser.add_argument(
        "-ptm",
        "--plot-statistical-maps",
        action="store_true",
        help="Plot statistical residual maps (single run)",
    )
    parser.add_argument(
        "-pr",
        "--plot-r-estimation",
        action="store_true",
        help="Plot R estimation for individual runs",
    )
    parser.add_argument(
        "-as",
        "--plot-all-spectra",
        action="store_true",
        help="Plot all spectra of the results",
    )
    parser.add_argument(
        "-ac",
        "--plot-all-cmb-recon",
        action="store_true",
        help="Plot all CMB reconstructions of the results",
    )
    parser.add_argument(
        "-asm",
        "--plot-all-systematic-maps",
        action="store_true",
        help="Plot systematic residual maps (multi-run mosaic)",
    )
    parser.add_argument(
        "-atm",
        "--plot-all-statistical-maps",
        action="store_true",
        help="Plot statistical residual maps (multi-run mosaic)",
    )
    parser.add_argument(
        "-ar",
        "--plot-all-r-estimation",
        action="store_true",
        help="Plot R estimation comparison across all runs",
    )
    # Aggregated grid/proxy relationships
    parser.add_argument(
        "-arc",
        "--plot-r-vs-c",
        action="store_true",
        help="Plot r vs number of clusters (per-parameter + total)",
    )
    parser.add_argument(
        "-avc",
        "--plot-v-vs-c",
        action="store_true",
        help="Plot variance vs number of clusters (per-parameter + total)",
    )
    parser.add_argument(
        "-arv",
        "--plot-r-vs-v",
        action="store_true",
        help="Plot r vs variance (per-parameter + total)",
    )
    parser.add_argument(
        "-am",
        "--plot-all-metrics",
        action="store_true",
        help="Plot metric distributions across runs (variance, NLL, sum Cl_BB)",
    )
    parser.add_argument(
        "-a",
        "--plot-all",
        action="store_true",
        help="Plot all results",
    )
    parser.add_argument(
        "-co",
        "--cache-only",
        action="store_true",
        help="Only compute and cache W_D_FG, skip all plotting",
    )
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="Force recomputation of cached W_D_FG values (use with --cache-only)",
    )
    parser.add_argument(
        "-cr",
        "--compute-residuals",
        type=str,
        choices=["all", "total", "statistical", "systematic", "none"],
        default="all",
        help="Which residuals to compute: all, total, statistical, systematic, or none",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        help="Directory to save snapshot data for incremental plotting",
    )
    parser.add_argument(
        "-ird",
        "--input-results-dir",
        type=str,
        nargs="*",
        default="results",
        help="Directory where results are stored",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["png", "pdf", "show"],
        default="png",
        help="Output format for plots: png (save as PNG), pdf (save as PDF), or show (display inline)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=14,
        help="Font size for all plot elements (default: 14)",
    )
    parser.add_argument(
        "-mi",
        "--max-iterations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--no-tex", action="store_true", help="Disable LaTeX rendering for plot text"
    )
    return parser.parse_args()
