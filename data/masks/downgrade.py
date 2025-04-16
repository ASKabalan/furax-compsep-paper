import healpy as hp
import jax
import numpy as np

mask = np.load("GAL_PlanckMasks_2048.npz")
mask = dict(mask)

nside = np.sqrt(mask["GAL060"].size / 12).astype(int)
npix = hp.nside2npix(nside)


for n in [64, 128, 256, 512]:
    mask_ud = jax.tree.map(lambda x: hp.ud_grade(x * 1.0, n).astype(np.uint8), mask)
    np.savez(f"GAL_PlanckMasks_{n}.npz", **mask_ud)
