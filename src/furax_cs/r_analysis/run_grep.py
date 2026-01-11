import os
import re

from ..logging_utils import info


def is_regex_token(token: str) -> bool:
    """Check if a single token contains regex metacharacters (not just OR syntax)."""
    groups = re.findall(r"\(([^)]+)\)", token)
    for group in groups:
        # Pure OR syntax: only contains pipe and alphanumeric
        if not re.match(r"^[\w|]+$", group):
            return True
    return False


def is_regex_pattern(pattern: str) -> bool:
    """Check if pattern contains any regex tokens."""
    tokens = pattern.split("_")
    return any(is_regex_token(token) for token in tokens)


def match_token_regex(folder_tokens: list[str], pattern_token: str) -> str | None:
    """Match pattern token against folder tokens using regex, return matched token or None."""
    try:
        regex = re.compile(f"^{pattern_token}$")
        for token in folder_tokens:
            if regex.match(token):
                return token
    except re.error:
        pass
    return None


def match_folder_with_regex_tokens(
    folder_tokens: list[str], pattern_tokens: list[str]
) -> tuple[bool, dict[int, str]]:
    """Match folder tokens against pattern tokens, some of which may be regex.

    Returns:
        (matched: bool, captures: dict mapping pattern token index to matched value)
    """
    captures = {}
    for i, pat_token in enumerate(pattern_tokens):
        if is_regex_token(pat_token):
            # Regex token: find a matching folder token
            matched = match_token_regex(folder_tokens, pat_token)
            if matched is None:
                return False, {}
            captures[i] = matched
        else:
            # Plain token or OR syntax: check if any option is in folder tokens
            if pat_token.startswith("(") and pat_token.endswith(")"):
                options = pat_token[1:-1].split("|")
                if not any(opt in folder_tokens for opt in options):
                    return False, {}
            else:
                if pat_token not in folder_tokens:
                    return False, {}
    return True, captures


def expand_pattern_with_captures(pattern_tokens: list[str], captures: dict[int, str]) -> str:
    """Build expanded pattern name by replacing regex tokens with captured values."""
    result_tokens = []
    for i, token in enumerate(pattern_tokens):
        if i in captures:
            result_tokens.append(captures[i])
        else:
            result_tokens.append(token)
    return "_".join(result_tokens)


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
) -> dict[str, tuple[list[str], int | tuple]]:
    """
    Search for result folders matching the given run specifications.

    Supports two matching modes:
    - Token mode: "kmeans_BD200" matches folders containing both tokens
    - Regex mode: "kmeans_BD(\\d+)" groups folders by captured values,
      creating separate entries like "kmeans_BD200", "kmeans_BD2500", etc.

    Parameters
    ----------
    result_folders : Union[str, list[str]]
        Directory or list of directories to scan for result folders.
    run_specs : list[str]
        List of keywords or keyword combinations to match.
        e.g., ["kmeans", "kmeans_abc", "kmeans_BD(\\d+)"].

    Returns
    -------
    dict[str, tuple[list[str], int | tuple]]
        Dictionary with run_spec (or expanded regex pattern) as key and
        tuple of (matching folders, index_spec) as value.
        e.g. {'kmeans_BD200': (['.../kmeans_BD200_...'], 0), ...}
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
        pattern_tokens = filter_str.split("_")

        if is_regex_pattern(filter_str):
            # Regex mode: group by captured values
            grouped = {}
            for path, folder_tokens in all_results.items():
                matched, captures = match_folder_with_regex_tokens(folder_tokens, pattern_tokens)
                if matched and captures:
                    expanded = expand_pattern_with_captures(pattern_tokens, captures)
                    grouped.setdefault(expanded, []).append(path)

            # Add each group as separate entry
            for expanded_name, paths in grouped.items():
                matches[expanded_name] = (paths, index_spec)
        else:
            # Token mode: existing logic
            filter_groups = parse_filter_kw(filter_str)
            matched_paths = []
            for path, tokens in all_results.items():
                if matches_filter(tokens, filter_groups):
                    matched_paths.append(path)
            matches[spec] = (matched_paths, index_spec)

    return matches
