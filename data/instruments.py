from pathlib import Path

import jax
import numpy as np
import yaml
from furax._instruments.sky import FGBusterInstrument


def get_instrument(instrument_name: str) -> FGBusterInstrument:
    """Get an instrument from its name."""
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
