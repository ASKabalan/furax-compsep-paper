import argparse
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["EQX_ON_ERROR"] = "nan"

import sys

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from furax.obs.stokes import Stokes
from jax_healpy import combine_masks

sys.path.append("../data")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark FGBuster and Furax Component Separation Methods"
    )

    parser.add_argument(
        "-n",
        "--nside",
        type=int,
        default=64,
        help="The nside of the map",
    )
    parser.add_argument(
        "-ns",
        "--noise-sim",
        type=int,
        default=200,
        help="Number of noise simulations",
    )

    parser.add_argument(
        "-r",
        "--runs",
        type=str,
        nargs="*",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    nside = args.nside
    noise_sims = args.noise_sim
    result_folder = "../results/"
    print("Loading data...")
    results = os.listdir(result_folder)
    filter_kw = [kw.split("_") for kw in args.runs]
    results_kw = {kw: kw.split("_") for kw in results}

    results_to_plot = []
    for result_name, res_kw in results_kw.items():
        for filt_kw in filter_kw:
            if all(kw in res_kw for kw in filt_kw):
                results_to_plot.append(result_name)
                break

    print("Results to plot: ", results_to_plot)

    results_to_plot = [f"{result_folder}{res}" for res in results_to_plot]

    PTEP_results = [res for res in results_to_plot if "PTEP" in res]
    comp_sep_results = [res for res in results_to_plot if "compsep" in res]

    PTEP_cmb_stokes, PTEP_cmb_recon = plot_PTEP(PTEP_results, nside, noise_sims)
    comsep_cmb_stokes, comsep_cmb_recon = plot_compsep(comp_sep_results, nside, noise_sims)

    if PTEP_cmb_stokes is None and comsep_cmb_stokes is None:
        print("Nothing to plot")
        return

    # Determine the best (least UNSEEN) input CMB map to display.
    if PTEP_cmb_stokes is not None and comsep_cmb_stokes is None:
        best_cmb_stokes = PTEP_cmb_stokes
    elif comsep_cmb_stokes is not None and PTEP_cmb_stokes is None:
        best_cmb_stokes = comsep_cmb_stokes
    elif PTEP_cmb_stokes is not None and comsep_cmb_stokes is not None:
        nb_unseen_ptep = jnp.sum(PTEP_cmb_stokes.q == hp.UNSEEN)
        nb_unseen_comsep = jnp.sum(comsep_cmb_stokes.q == hp.UNSEEN)
        best_cmb_stokes = (
            PTEP_cmb_stokes if nb_unseen_ptep < nb_unseen_comsep else comsep_cmb_stokes
        )

    # Adjust subplot grid based on availability.
    if PTEP_cmb_recon is None or comsep_cmb_recon is None:
        subs = (2, 2)
    else:
        subs = (2, 3)

    _ = plt.figure(figsize=jax.tree.map(lambda x: x * 4, subs))
    axis_indx = 1

    if comsep_cmb_recon is not None:
        hp.mollview(
            comsep_cmb_recon.q,
            title="CompSep CMB Reconstruction",
            sub=subs + (axis_indx,),
            bgcolor=(0.0,) * 4,
        )
        axis_indx += 1

    if PTEP_cmb_recon is not None:
        hp.mollview(
            PTEP_cmb_recon.q,
            title="PTEP CMB Reconstruction",
            sub=subs + (axis_indx,),
            bgcolor=(0.0,) * 4,
        )
        axis_indx += 1

    hp.mollview(
        best_cmb_stokes.q, title="Input CMB Map", sub=subs + (axis_indx,), bgcolor=(0.0,) * 4
    )
    axis_indx += 1

    if comsep_cmb_recon is not None:
        hp.mollview(
            comsep_cmb_recon.u,
            title="CompSep CMB Reconstruction (U)",
            sub=subs + (axis_indx,),
            bgcolor=(0.0,) * 4,
        )
        axis_indx += 1

    if PTEP_cmb_recon is not None:
        hp.mollview(
            PTEP_cmb_recon.u,
            title="PTEP CMB Reconstruction (U)",
            sub=subs + (axis_indx,),
            bgcolor=(0.0,) * 4,
        )
        axis_indx += 1

    hp.mollview(
        best_cmb_stokes.u, title="Input CMB Map (U)", sub=subs + (axis_indx,), bgcolor=(0.0,) * 4
    )
    plt.show()


def plot_compsep(comsep_results, nside, noise_sims):
    if len(comsep_results) == 0:
        print("No compsep results")
        return None, None

    cmb_recons = []
    cmb_maps = []
    masks = []
    indices_list = []

    for res_folder in comsep_results:
        run_data = dict(np.load(f"{res_folder}/results.npz"))
        cmb_map = np.load(f"{res_folder}/best_params.npz")["I_CMB"]
        mask = np.load(f"{res_folder}/mask.npy")
        (indices,) = jnp.where(mask == 1)

        cmb_recon = run_data["CMB_O"]
        cmb_recon = Stokes.from_stokes(Q=cmb_recon[:, 0], U=cmb_recon[:, 1])
        cmb_recon_mean = jax.tree.map(lambda x: x.mean(axis=0), cmb_recon)

        cmb_recons.append(cmb_recon_mean)
        cmb_maps.append(cmb_map)
        masks.append(mask)
        indices_list.append(indices)

    for cmb_map in cmb_maps:
        assert (cmb_map == cmb_maps[0]).all(), "CMB maps are not the same"

    full_mask = np.zeros_like(masks[0])
    for mask in masks:
        full_mask = np.logical_or(full_mask, mask)

    (full_mask_indices,) = jnp.where(full_mask == 1)
    combined_cmb_recon = combine_masks(cmb_recons, indices_list, nside)

    cmb_map = cmb_maps[0]
    cmb_stokes = Stokes.from_stokes(Q=cmb_map[1], U=cmb_map[2])
    cmb_stokes = jax.tree.map(lambda x: np.where(full_mask == 1, x, hp.UNSEEN), cmb_stokes)

    def mse(a, b):
        seen_x = jax.tree.map(lambda x: x[x != hp.UNSEEN], a)
        seen_y = jax.tree.map(lambda x: x[x != hp.UNSEEN], b)
        return jax.tree.map(lambda x, y: jnp.mean((x - y) ** 2), seen_x, seen_y)

    mse_cmb = mse(combined_cmb_recon, cmb_stokes)
    cmb_recon_var = jax.tree.map(lambda x: jnp.var(x[x != hp.UNSEEN]), combined_cmb_recon)
    cmb_input_var = jax.tree.map(lambda x: jnp.var(x[x != hp.UNSEEN]), cmb_stokes)
    print("COMP SEP RESULTS")
    print("======================")
    print(f"MSE CMB: {mse_cmb}")
    print(f"Reconstructed CMB variance: {cmb_recon_var}")
    print(f"Input CMB variance: {cmb_input_var}")
    print("======================")

    _ = plt.figure(figsize=(8, 8))
    hp.mollview(
        combined_cmb_recon.q,
        title="CompSep Reconstructed CMB (Q)",
        sub=(2, 2, 1),
        bgcolor=(0.0,) * 4,
    )
    hp.mollview(cmb_stokes.q, title="Input CMB Map (Q)", sub=(2, 2, 2), bgcolor=(0.0,) * 4)
    hp.mollview(
        combined_cmb_recon.u,
        title="CompSep Reconstructed CMB (U)",
        sub=(2, 2, 3),
        bgcolor=(0.0,) * 4,
    )
    hp.mollview(cmb_stokes.u, title="Input CMB Map (U)", sub=(2, 2, 4), bgcolor=(0.0,) * 4)
    plt.show()

    return cmb_stokes, combined_cmb_recon


def plot_PTEP(PTEP_results, nside, noise_sims):
    if len(PTEP_results) == 0:
        print("No PTEP results")
        return None, None

    cmb_recons = []
    cmb_maps = []
    masks = []
    indices_list = []

    for res_folder in PTEP_results:
        run_data = dict(np.load(f"{res_folder}/results.npz"))
        cmb_map = np.load(f"{res_folder}/best_params.npz")["I_CMB"]
        mask = np.load(f"{res_folder}/mask.npy")
        (indices,) = jnp.where(mask == 1)

        cmb_recon = run_data["CMB_O"]
        cmb_recon = Stokes.from_stokes(Q=cmb_recon[:, 0], U=cmb_recon[:, 1])
        cmb_recon_mean = jax.tree.map(lambda x: x.mean(axis=0), cmb_recon)
        cmb_map_stokes = Stokes.from_stokes(Q=cmb_map[0], U=cmb_map[1])
        print(f"STOKEs shape of cmb_map: {cmb_map_stokes.structure}")
        print(f"STOKEs shape of recon: {cmb_recon_mean.structure}")

        cmb_recons.append(cmb_recon_mean)
        cmb_maps.append(cmb_map_stokes)
        masks.append(mask)
        indices_list.append(indices)

    full_mask = np.zeros_like(masks[0])
    for mask in masks:
        full_mask = np.logical_or(full_mask, mask)

    (full_mask_indices,) = jnp.where(full_mask == 1)
    combined_cmb_recon = combine_masks(cmb_recons, indices_list, nside)
    cmb_stokes = combine_masks(cmb_maps, indices_list, nside)

    def mse(a, b):
        seen_x = jax.tree.map(lambda x: x[x != hp.UNSEEN], a)
        seen_y = jax.tree.map(lambda x: x[x != hp.UNSEEN], b)
        return jax.tree.map(lambda x, y: jnp.mean((x - y) ** 2), seen_x, seen_y)

    mse_cmb = mse(combined_cmb_recon, cmb_stokes)
    cmb_recon_var = jax.tree.map(lambda x: jnp.var(x[x != hp.UNSEEN]), combined_cmb_recon)
    cmb_input_var = jax.tree.map(lambda x: jnp.var(x[x != hp.UNSEEN]), cmb_stokes)
    print("PTEP RESULTS")
    print("======================")
    print(f"MSE CMB: {mse_cmb}")
    print(f"Reconstructed CMB variance: {cmb_recon_var}")
    print(f"Input CMB variance: {cmb_input_var}")
    print("======================")

    _ = plt.figure(figsize=(12, 8))
    hp.mollview(
        combined_cmb_recon.q, title="PTEP Reconstructed CMB (Q)", sub=(2, 3, 1), bgcolor=(0.0,) * 4
    )
    hp.mollview(cmb_stokes.q, title="Input CMB Map (Q)", sub=(2, 3, 2), bgcolor=(0.0,) * 4)
    hp.mollview(
        np.abs(combined_cmb_recon.q - cmb_stokes.q),
        title="Difference (Q)",
        sub=(2, 3, 3),
        bgcolor=(0.0,) * 4,
    )
    hp.mollview(
        combined_cmb_recon.u, title="PTEP Reconstructed CMB (U)", sub=(2, 3, 4), bgcolor=(0.0,) * 4
    )
    hp.mollview(cmb_stokes.u, title="Input CMB Map (U)", sub=(2, 3, 5), bgcolor=(0.0,) * 4)
    hp.mollview(
        np.abs(combined_cmb_recon.u - cmb_stokes.u),
        title="Difference (U)",
        sub=(2, 3, 6),
        bgcolor=(0.0,) * 4,
    )
    plt.show()

    return cmb_stokes, combined_cmb_recon


if __name__ == "__main__":
    main()
