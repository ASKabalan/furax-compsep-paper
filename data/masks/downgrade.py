
import numpy as np
import healpy as hp
import jax

mask = np.load('GAL_PlanckMasks_2048.npz')
mask = dict(mask)

nside = np.sqrt(mask['GAL060'].size/12).astype(int)
npix = hp.nside2npix(nside)


for n in [64 , 128 , 256 , 512]:

    ipix = np.arange(npix)
    ipix_ud = hp.ud_grade(ipix, n)

    mask_ud = jax.tree.map(lambda x: x[ipix_ud], mask)
    np.savez(f'GAL_PlanckMasks_{n}.npz', **mask_ud)