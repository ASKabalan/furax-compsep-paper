from pathlib import Path

import jax
import numpy as np
import yaml
from furax._instruments.sky import FGBusterInstrument


def get_instrument(instrument_name: str) -> FGBusterInstrument:
    """Get an instrument configuration by name.

    Args:
        instrument_name: Name of the instrument (e.g., "LiteBIRD", "Planck").
            Must correspond to an entry in `instruments.yaml`.
            Use "default" for the FGBuster default instrument.

    Returns:
        The instrument configuration object with frequency bands and sensitivities.

    Raises:
        ValueError: If `instrument_name` is not found in the configuration.
    """
    current_dir = Path(__file__).parent
    with open(f"{current_dir}/instruments.yaml") as f:
        instruments = yaml.safe_load(f)

    if instrument_name == "default":
        instrument = FGBusterInstrument.default_instrument()
        return jax.tree.map(np.array, instrument, is_leaf=lambda x: isinstance(x, list))

    if instrument_name not in instruments:
        raise ValueError(f"Unknown instrument {instrument_name}.")

    instrument_yaml = instruments[instrument_name]
    frequency = instrument_yaml["frequency"]
    depth_i = instrument_yaml["depth_i"]
    depth_p = instrument_yaml["depth_p"]

    instrument = FGBusterInstrument(frequency, depth_i, depth_p)
    return jax.tree.map(np.array, instrument, is_leaf=lambda x: isinstance(x, list))
