"""Utilities for loading and managing grid search space configurations."""

from pathlib import Path

import jax.numpy as jnp
import yaml
from jaxtyping import Array


def load_search_space(filepath: str | Path | None = None) -> dict:
    """Load search space configuration from YAML file.

    Parameters
    ----------
    filepath : str, Path, or None
        Path to custom search space YAML file. If None, loads the default
        configuration from search_spaces_default.yaml.

    Returns
    -------
    dict
        Dictionary with JAX arrays for T_d_patches, B_d_patches, B_s_patches.

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    ValueError
        If the YAML file is missing required keys or has invalid values.
    """
    if filepath is None:
        # Load default configuration from data directory
        data_dir = Path(__file__).parent
        filepath = data_dir / "search_spaces_default.yaml"
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Search space file not found: {filepath}")

    with open(filepath) as f:
        config = yaml.safe_load(f)

    # Validate required keys
    required_keys = ["T_d_patches", "B_d_patches", "B_s_patches"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Search space YAML missing required keys: {missing_keys}. "
            f"Required keys: {required_keys}"
        )

    # Convert to JAX arrays
    search_space = search_space_to_jax(config)

    return search_space


def search_space_to_jax(config: dict) -> dict:
    """Convert search space configuration from YAML to JAX arrays.

    Parameters
    ----------
    config : dict
        Dictionary with search space parameters as lists or numpy arrays.

    Returns
    -------
    dict
        Dictionary with JAX arrays ready for grid search.
    """
    search_space = {}

    # Convert T_d_patches
    if "T_d_patches" in config:
        search_space["T_d_patches"] = jnp.array(config["T_d_patches"])

    # Convert B_d_patches - handle both list and range specifications
    if "B_d_patches" in config:
        b_d = config["B_d_patches"]
        if isinstance(b_d, list):
            search_space["B_d_patches"] = jnp.array(b_d)
        else:
            # Handle potential dict specification for ranges
            search_space["B_d_patches"] = jnp.array(b_d)

    # Convert B_s_patches
    if "B_s_patches" in config:
        search_space["B_s_patches"] = jnp.array(config["B_s_patches"])

    return search_space


def dump_default_search_space(output_path: str | Path) -> None:
    """Dump the default search space configuration to a YAML file.

    This creates a template file that users can customize for their needs.

    Parameters
    ----------
    output_path : str or Path
        Path where the default search space YAML will be saved.

    Raises
    ------
    IOError
        If the file cannot be written.
    """
    output_path = Path(output_path)

    # Load the default configuration
    data_dir = Path(__file__).parent
    default_path = data_dir / "search_spaces_default.yaml"

    if not default_path.exists():
        raise FileNotFoundError(
            f"Default search space file not found at {default_path}. "
            "This should not happen - please check the package installation."
        )

    # Copy the default file to the output location
    with open(default_path) as f:
        default_config = f.read()

    with open(output_path, "w") as f:
        f.write(default_config)

    print(f"Default search space configuration saved to: {output_path}")
    print("You can now edit this file to customize the search space.")


def validate_search_space(search_space: dict) -> None:
    """Validate that search space has valid structure and values.

    Parameters
    ----------
    search_space : dict
        Dictionary with JAX arrays for search space parameters.

    Raises
    ------
    ValueError
        If validation fails.
    """
    required_keys = ["T_d_patches", "B_d_patches", "B_s_patches"]

    for key in required_keys:
        if key not in search_space:
            raise ValueError(f"Search space missing required key: {key}")

        arr = search_space[key]
        if not isinstance(arr, Array):
            raise ValueError(f"{key} must be an array, got {type(arr)}")

        if arr.size == 0:
            raise ValueError(f"{key} cannot be empty")

        if jnp.any(arr < 1):
            raise ValueError(f"{key} values must be >= 1, got minimum {jnp.min(arr)}")
