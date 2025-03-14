import yaml
from furax._instruments.sky import FGBusterInstrument


def get_instrument(instrument_name: str) -> FGBusterInstrument:
    """Get an instrument from its name."""
    with open("instrument.yaml") as f:
        instruments = yaml.safe_load(f)

    if instrument_name not in instruments:
        raise ValueError(f"Unknown instrument {instrument_name}.")

    instrument_yaml = instruments[instrument_name]
    frequency = instrument_yaml["frequency"]
    depth_i = instrument_yaml["depth_i"]
    depth_p = instrument_yaml["depth_p"]

    return FGBusterInstrument(frequency, depth_i, depth_p)
