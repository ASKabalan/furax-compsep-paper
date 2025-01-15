
import jax
import healpy as hp
import numpy as np

from furax.comp_sep import spectral_cmb_variance, get_clusters, optimize

import jax.numpy as jnp

from furax._base.core import HomothetyOperator
from furax.landscapes import StokesPyTree
from furax.tree import allclose
from functools import partial
import matplotlib.pyplot as plt


GAL020 = np.load('GAL_PlanckMasks_64.npz')['GAL020']
GAL040 = np.load('GAL_PlanckMasks_64.npz')['GAL040']
GAL060 = np.load('GAL_PlanckMasks_64.npz')['GAL060']


temp_dust_patches_count = 10
beta_dust_patches_count = 20
beta_pl_patches_count = 5

indices , = jnp.where(GAL020 > 0)
temp_dust_patch_indices = get_clusters(GAL020, indices , temp_dust_patches_count, jax.random.PRNGKey(0))

figure = plt.figure(figsize=(15, 5))
hp.mollview(GAL020, title='GAL020' , sub=(1 , 2 , 1))
hp.mollview(temp_dust_patch_indices, title='Clusters' , sub=(1 , 2 , 2))
plt.show()