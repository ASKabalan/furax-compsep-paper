import os

from .logging_utils import info


def parse_run_spec(run_spec):
    """Parse a run spec string into filter and index information."""
    if "," not in run_spec:
        return run_spec, 0

    filter_part, index_part = run_spec.rsplit(",", 1)
    index_part = index_part.strip()

    if "-" in index_part:
        start, end = index_part.split("-", 1)
        return filter_part, (int(start.strip()), int(end.strip()))
    else:
        return filter_part, int(index_part)


def parse_filter_kw(kw_string):
    """Split run keywords into AND-of-OR groups for matching."""
    groups = kw_string.split("_")
    parsed = []
    for group in groups:
        if group.startswith("(") and group.endswith(")"):
            options = group[1:-1].split("|")
            parsed.append(set(options))
        else:
            parsed.append({group})
    return parsed


def matches_filter(name_parts, filter_groups):
    """Return True if the keyword groups all match the provided name parts."""
    return all(any(option in name_parts for option in group) for group in filter_groups)


def run_grep(
    result_folders: str | list[str],
    run_specs: list[str],
) -> dict[str, list[str]]:
    """
    Search for result folders matching the given run specifications.

    Parameters
    ----------
    result_folders : Union[str, list[str]]
        Directory or list of directories to scan for result folders.
    run_specs : list[str]
        List of keywords or keyword combinations to match.
        e.g., ["kmeans", "kmeans_abc"].

    Returns
    -------
    dict[str, list[str]]
        Dictionary with run_spec as key and list of matching folders as value.
        e.g. {'kmeans': ['.../kmeans_abc'], ...}
    """
    if isinstance(result_folders, str):
        result_folders = [result_folders]

    # 1. Scan for all potential result folders
    all_results = {}
    for folder in result_folders:
        info(f"Scanning results folder: {folder}")
        if not os.path.exists(folder):
            raise ValueError(f"Results folder '{folder}' does not exist.")

        for root, dirs, files in os.walk(folder):
            if not dirs:  # leaf directory
                info(f" -> Handling subfolder: {root}")
                name = os.path.basename(root)
                # Tokenize by underscore for matching
                tokens = name.split("_")
                all_results[root] = tokens

    # 2. Match specs
    matches = {}
    for spec in run_specs:
        filter_str, index_spec = parse_run_spec(spec)
        filter_groups = parse_filter_kw(filter_str)
        matched_paths = []
        for path, tokens in all_results.items():
            if matches_filter(tokens, filter_groups):
                matched_paths.append(path)
        matches[spec] = (matched_paths, index_spec)

    return matches
