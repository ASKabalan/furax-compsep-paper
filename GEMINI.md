I want to make a prompt for CLAUDE Code to implement a package following UML diagram

UML(1).png

# Context 

I have already launched a claude code and it implemented something that was not what I wanted.
the package exists in folder `src/` and the package name is `furax-cs`.

some things about the UML : 

instrument corresponds to file `data/instruments.py`
sky_signal corresponds to file `data/generate_maps.py` which generates and caches maps

CompenentSeparationConfig `STATIC DATA CLASS` and you can created by following 

```py
import jax
from dataclasses import dataclass, field
@jax.tree_util.register_dataclass
@dataclass
class StaticDataClass:
  op: str = field(metadata=dict(static=True))  # marked as static meta field.

@jax.tree_util.register_dataclass
@dataclass
class DynamicDataClass:
  x: Array
```

ClusteringStrategy has an abstract create_patches method that is implemented by the subclasses.
This method takes only 3 ints one per parameter `beta_dust`, `beta_pl`, and `temp_dust`.

This should be called each time in another function that creates the patches.
You should like in scripts 
content/08-KMeans-model.py
content/05-PTEP-model.py


to see how the patches are created. 

## KMeans patche creation

```python
n_regions = {
    "temp_dust_patches": T_d_patches,
    "beta_dust_patches": B_d_patches,
    "beta_pl_patches": B_s_patches,
}

patch_indices = jax.tree.map(
    lambda c, mp: get_clusters(
        mask, indices, c, jax.random.key(0), max_centroids=mp, initial_sample_size=1
    ),
    n_regions,
    max_patches,
)
guess_clusters = get_cutout_from_mask(patch_indices, indices)
# Normalize the cluster to make indexing more logical
guess_clusters = jax.tree.map(
    lambda g, c, mp: normalize_by_first_occurrence(g, c, mp).astype(jnp.int64),
    guess_clusters,
    n_regions,
    max_patches,
)
guess_clusters = jax.tree.map(lambda x: x.astype(jnp.int64), guess_clusters)
```

## MultiRes patches creation

```py
ud_beta_d = int(args.target_ud_grade[0])
ud_temp_d = int(args.target_ud_grade[1])
ud_beta_pl = int(args.target_ud_grade[2])
# Create a dummy full-sky map (of ones) to generate the downg
npix = nside**2 * 12
ipix = np.arange(npix)
def ud_grade(ipix, nside_in, nside_out):
    if nside_out == 0:
        return np.zeros_like(ipix)
    else:
        lowered = hp.ud_grade(ipix, nside_out=nside_out)
        return hp.ud_grade(lowered, nside_out=nside_in)
ud_beta_d_map = ud_grade(ipix, nside, ud_beta_d)
ud_temp_d_map = ud_grade(ipix, nside, ud_temp_d)
ud_beta_pl_map = ud_grade(ipix, nside, ud_beta_pl)
# These downgraded maps serve as our patch indices.
patch_indices = {
    "beta_dust_patches": ud_beta_d_map,
    "temp_dust_patches": ud_temp_d_map,
    "beta_pl_patches": ud_beta_pl_map,
}
patch_indices = get_cutout_from_mask(patch_indices, indices)
```

I have function in jax_healpy.ud_grade that work exactly like healpy.ud_grade.
Use it instead of healpy.ud_grade.

## Base compsep

for the baseclass
there should be another function that computes_results using the patches and looks like compute function in src/furax_cs/component_separation/base.py

this compute function can take only number of patches and then creates the patches using the clustering strategy and then computes the results.

in the run function you should do the following:

create patches
compute results
results is the state that is a dynamic data class

## Other modules

r_estimations should allow compute function as implemented in `content/09-R_estimation.py`

plotting does the plotting like someof the function in the R_estimation script as well `content/09-R_estimation.py`

sky_signal should be able to generate maps like in `data/generate_maps.py`
instrument corresponds to file `data/instruments.py`

WARNING YOU ARE NOT SUPPOSED TO IMPLEMENT
please scan this repo and and find how good is this prompt to implement this function
previously, claude had too much freedom and implemented something that was not what I wanted.
I also want to say that the package that exists in `src/furax_cs` is not what I want, so you claude can completely change it.

please refine and create a prompt and write it into PROMPT.md file in the root of the repo.