"""Logging utilities for r_analysis package."""

import sys
import warnings
from contextlib import contextmanager
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    @classmethod
    def is_tty(cls):
        """Check if stdout is a TTY (supports colors)."""
        return sys.stdout.isatty()

    @classmethod
    def disable(cls):
        """Disable all colors."""
        cls.BLUE = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.RED = ""
        cls.CYAN = ""
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""


# Disable colors if not in a TTY
if not Colors.is_tty():
    Colors.disable()


def info(message):
    """Print an informational message.

    Parameters
    ----------
    message : str
        Message to print
    """
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {message}")


def success(message):
    """Print a success message with checkmark.

    Parameters
    ----------
    message : str
        Message to print
    """
    print(f"{Colors.GREEN}✓{Colors.RESET} {message}")


def warning(message):
    """Print a warning message.

    Parameters
    ----------
    message : str
        Warning message to print
    """
    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {message}")


def error(message):
    """Print an error message.

    Parameters
    ----------
    message : str
        Error message to print
    """
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {message}", file=sys.stderr)


def hint(message):
    """Print a hint message for user guidance.

    Parameters
    ----------
    message : str
        Hint message to print
    """
    print(f"{Colors.CYAN}[HINT]{Colors.RESET} {message}")


def debug(message):
    """Print a debug message (dimmed).

    Parameters
    ----------
    message : str
        Debug message to print
    """
    print(f"{Colors.DIM}[DEBUG] {message}{Colors.RESET}")


def banner(message, char="=", width=60):
    """Print a banner message.

    Parameters
    ----------
    message : str
        Message to display in banner
    char : str
        Character to use for banner lines
    width : int
        Width of the banner
    """
    print(char * width)
    print(message)
    print(char * width)


def format_cache_summary(snapshot_store, titles_to_plot):
    """Format a summary of snapshot cache status.

    Parameters
    ----------
    snapshot_store : dict
        Dictionary of cached snapshot entries
    titles_to_plot : list
        List of all run titles to plot

    Returns
    -------
    str
        Formatted summary message
    """
    cached = sum(1 for title in titles_to_plot if title in snapshot_store)
    total = len(titles_to_plot)
    if cached == total:
        return f"All {total} runs cached in snapshot"
    elif cached == 0:
        return f"No runs cached (will compute all {total})"
    else:
        return f"{cached}/{total} runs cached in snapshot ({total - cached} to compute)"


def format_folder_summary(results_to_plot):
    """Format a compact summary of folders to process.

    Parameters
    ----------
    results_to_plot : list of list
        Nested list of result folder paths

    Returns
    -------
    str
        Formatted summary message
    """
    total_folders = sum(len(group) for group in results_to_plot)
    num_groups = len(results_to_plot)
    return f"Found {num_groups} run group{'s' if num_groups != 1 else ''} ({total_folders} total folders)"


def compact_path(path, max_length=50):
    """Shorten a path for display if it's too long.

    Parameters
    ----------
    path : str
        Path to shorten
    max_length : int
        Maximum length before shortening

    Returns
    -------
    str
        Shortened path with ellipsis if needed
    """
    if len(path) <= max_length:
        return path
    path_obj = Path(path)
    name = path_obj.name
    if len(name) <= max_length - 10:
        return f".../{name}"
    return f"...{path[-max_length:]}"


def format_residual_flags(compute_syst, compute_stat, compute_total):
    """Format residual computation flags into a readable message.

    Parameters
    ----------
    compute_syst : bool
        Whether computing systematic residuals
    compute_stat : bool
        Whether computing statistical residuals
    compute_total : bool
        Whether computing total residuals

    Returns
    -------
    str
        Formatted message describing what will be computed
    """
    components = []
    if compute_syst:
        components.append("systematic")
    if compute_stat:
        components.append("statistical")
    if compute_total and not (compute_syst and compute_stat):
        components.append("total")

    if not components:
        return "No residuals"

    return "Computing: " + ", ".join(components) + " residuals"


@contextmanager
def suppress_runtime_warnings():
    """Context manager to suppress specific runtime warnings.

    Suppresses RuntimeWarning about invalid values in logarithm operations,
    which can occur during r estimation when cl_model has zeros.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in log")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        yield


def format_r_result(r_best, sigma_r_neg, sigma_r_pos):
    """Format r estimation result for display.

    Parameters
    ----------
    r_best : float or None
        Best-fit r value
    sigma_r_neg : float or None
        Negative error bar
    sigma_r_pos : float or None
        Positive error bar

    Returns
    -------
    str or None
        Formatted result string, or None if r estimation not available
    """
    if r_best is None:
        return None

    if sigma_r_neg is not None and sigma_r_pos is not None:
        avg_sigma = (abs(sigma_r_neg) + sigma_r_pos) / 2
        return f"r = {r_best:.4f} ± {avg_sigma:.4f}"
    else:
        return f"r = {r_best:.4f}"
