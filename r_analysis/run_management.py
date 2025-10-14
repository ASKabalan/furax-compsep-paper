def parse_run_spec(run_spec):
    """Parse a run spec string into filter and index information."""
    if "," not in run_spec:
        return run_spec, None

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


def expand_run_specs(run_specs, titles):
    """Expand run specifications into (filter, index, title) tuples."""
    expanded = []

    for run_spec, base_title in zip(run_specs, titles):
        filter_str, index_spec = parse_run_spec(run_spec)

        if index_spec is None:
            expanded.append([(filter_str, 0, base_title)])
        elif isinstance(index_spec, int):
            expanded.append([(filter_str, index_spec, base_title)])
        elif isinstance(index_spec, tuple):
            start, end = index_spec
            group = []
            for idx in range(start, end + 1):
                title = f"{base_title} ({idx})"
                group.append((filter_str, idx, title))
            expanded.append(group)
        else:
            raise ValueError(f"Unknown index specification: {index_spec}")

    return expanded
