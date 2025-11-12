import argparse
import os
from pathlib import Path

import healpy as hp
import jax
import numpy as np

current_folder = Path(__file__).parent


def generate_masks(nside_list=None):
    """Downgrade Planck galactic masks to lower HEALPix resolutions.

    Parameters
    ----------
    nside_list : list of int, optional
        Target resolutions for downgraded masks (default: [64, 128, 256, 512]).

    Notes
    -----
    Reads GAL_PlanckMasks_2048.npz and generates downgraded versions for each
    target nside, saved to masks/ directory.
    """
    if nside_list is None:
        nside_list = [64, 128, 256, 512]

    mask = np.load(f"{current_folder}/GAL_PlanckMasks_2048.npz")
    mask = dict(mask)
    mask_folder = "masks/"
    os.makedirs(mask_folder, exist_ok=True)

    for n in nside_list:
        mask_ud = jax.tree.map(lambda x: hp.ud_grade(x * 1.0, n).astype(np.uint8), mask)
        np.savez(f"{mask_folder}/GAL_PlanckMasks_{n}.npz", **mask_ud)
        print(f"Generated mask for nside {n}")


def main():
    parser = argparse.ArgumentParser(
        description="Downgrade Planck galactic masks to lower resolutions"
    )
    parser.add_argument(
        "--nside",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="Target HEALPix resolution(s) for downgraded masks (default: 64 128 256 512)",
    )

    args = parser.parse_args()

    generate_masks(nside_list=args.nside)


if __name__ == "__main__":
    main()
