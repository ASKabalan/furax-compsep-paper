import os

os.environ["EQX_ON_ERROR"] = "nan"
import os

import healpy as hp
import jax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax_healpy import from_cutout_to_fullmap, get_clusters, get_cutout_from_mask


def filter_constant_param(input_dict, indx):
    out_dict = {}
    out_dict["B_d_patches"] = input_dict["B_d_patches"][indx]
    out_dict["T_d_patches"] = input_dict["T_d_patches"][indx]
    out_dict["B_s_patches"] = input_dict["B_s_patches"][indx]
    out_dict["value"] = input_dict["value"][indx]
    out_dict["NLL"] = input_dict["NLL"][indx]

    out_dict["beta_dust"] = input_dict["beta_dust"][indx][: out_dict["B_d_patches"]]
    out_dict["temp_dust"] = input_dict["temp_dust"][indx][: out_dict["T_d_patches"]]
    out_dict["beta_pl"] = input_dict["beta_pl"][indx][: out_dict["B_s_patches"]]
    return out_dict


def plot_cmb_nll_vs_B_d_patches_with_noise(results, best_params, out_folder):
    sns.set_context("paper")

    B_d_patches = results["B_d_patches"]  # dust patch count (x-axis)
    cmb_variance_mean = np.mean(results["value"], axis=1)  # Mean CMB variance
    cmb_variance_std = np.std(
        results["value"], axis=1
    )  # Variance (std) of CMB variance
    nll_mean = np.mean(results["NLL"], axis=1)  # Mean Negative log-likelihood
    nll_std = np.std(results["NLL"], axis=1)  # Variance (std) of NLL

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot CMB variance vs. B_d_patches with error bars
    axs[0].errorbar(
        B_d_patches,
        cmb_variance_mean,
        yerr=cmb_variance_std,
        fmt="o",
        color="blue",
        label="Grid Search",
    )
    axs[0].axhline(
        y=best_params["value"], color="red", linestyle="--", label="Best CMB Variance"
    )
    axs[0].axvline(
        x=best_params["B_d_patches"],
        color="orange",
        linestyle="--",
        label="Best B_d_patches",
    )
    axs[0].set_xlabel("B_d_patches (dust patch count)")
    axs[0].set_ylabel("CMB Variance")
    axs[0].set_title("CMB Variance vs. B_d_patches")
    axs[0].legend()

    # Plot NLL vs. B_d_patches with error bars
    axs[1].errorbar(
        B_d_patches, nll_mean, yerr=nll_std, fmt="o", color="green", label="Grid Search"
    )
    axs[1].axhline(y=best_params["NLL"], color="red", linestyle="--", label="Best NLL")
    axs[1].axvline(
        x=best_params["B_d_patches"],
        color="orange",
        linestyle="--",
        label="Best B_d_patches",
    )
    axs[1].set_xlabel("B_d_patches (dust patch count)")
    axs[1].set_ylabel("Negative Log Likelihood")
    axs[1].set_title("NLL vs. B_d_patches")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"{out_folder}/cmb_nll_vs_B_d_patches_with_errorbars.png",
        dpi=600,
        transparent=True,
    )


def plot_healpix_projection_with_noise(mask, nside, results, best_params, out_folder):
    sns.set_context("paper")

    (indices,) = jnp.where(mask == 1)

    # Get best run
    best_run = filter_constant_param(results, 0)
    patches = get_clusters(
        mask,
        indices,
        best_run["B_d_patches"],
        jax.random.key(0),
        max_centroids=best_run["B_d_patches"],
    ).astype(jnp.int32)
    patches = get_cutout_from_mask(patches, indices)

    best_spectral_params = best_params["beta_dust"][patches]
    best_healpix_map = from_cutout_to_fullmap(best_spectral_params, indices, nside)

    # Process all runs to compute mean & std deviation
    beta_dust_values = best_run["beta_dust"]

    mean_spectral_params = jnp.mean(beta_dust_values, axis=0)[patches]
    std_spectral_params = jnp.std(beta_dust_values, axis=0)[patches]

    mean_healpix_map = from_cutout_to_fullmap(mean_spectral_params, indices, nside)
    std_dev_map = from_cutout_to_fullmap(std_spectral_params, indices, nside)

    # Plot results
    plt.figure(figsize=(6, 12))
    hp.mollview(
        best_healpix_map, title="Best Beta Dust Map", sub=(3, 1, 1), bgcolor=(0.0,) * 4
    )
    hp.mollview(
        mean_healpix_map, title="Mean Beta Dust Map", sub=(3, 1, 2), bgcolor=(0.0,) * 4
    )
    hp.mollview(
        std_dev_map,
        title="Standard Deviation (Uncertainty)",
        sub=(3, 1, 3),
        bgcolor=(0.0,) * 4,
    )

    plt.tight_layout()
    plt.savefig(f"{out_folder}/beta_dust_projection.png", dpi=600, transparent=True)


def plot_cmb_nll_vs_B_d_patches(results, best_params, out_folder):
    sns.set_context("paper")
    # Extract values from the grid search results
    B_d_patches = results["B_d_patches"]  # dust patch count (x-axis)
    cmb_variance = results["value"]  # CMB variance values from grid search
    nll = results["NLL"]  # Negative log-likelihood values

    # Create subplots: one for CMB variance, one for NLL
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot CMB variance vs. B_d_patches
    axs[0].scatter(B_d_patches, cmb_variance, color="blue", label="Grid Search")
    axs[0].axhline(
        y=best_params["value"], color="red", linestyle="--", label="Best CMB Variance"
    )
    axs[0].axvline(
        x=best_params["B_d_patches"],
        color="orange",
        linestyle="--",
        label="Best B_d_patches",
    )
    axs[0].set_xlabel("B_d_patches (dust patch count)")
    axs[0].set_ylabel("CMB Variance")
    axs[0].set_title("CMB Variance vs. B_d_patches")
    axs[0].legend()

    # Plot NLL vs. B_d_patches
    axs[1].scatter(B_d_patches, nll, color="green", label="Grid Search")
    axs[1].axhline(y=best_params["NLL"], color="red", linestyle="--", label="Best NLL")
    axs[1].axvline(
        x=best_params["B_d_patches"],
        color="orange",
        linestyle="--",
        label="Best B_d_patches",
    )
    axs[1].set_xlabel("B_d_patches (dust patch count)")
    axs[1].set_ylabel("Negative Log Likelihood")
    axs[1].set_title("NLL vs. B_d_patches")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"{out_folder}/cmb_nll_vs_B_d_patches.png",
        dpi=600,
        transparent=True,
    )


def plot_healpix_projection(mask, nside, results, best_params, out_folder):
    sns.set_context("paper")

    (indices,) = jnp.where(mask == 1)
    best_run = filter_constant_param(results, 0)
    patches = get_clusters(
        mask,
        indices,
        best_run["B_d_patches"],
        jax.random.key(0),
        max_centroids=best_run["B_d_patches"],
    ).astype(jnp.int32)
    patches = get_cutout_from_mask(patches, indices)
    best_spectral_params = best_params["beta_dust"][patches]
    result_spectral_params = best_run["beta_dust"][patches]

    best_healpix_map = from_cutout_to_fullmap(best_spectral_params, indices, nside)
    result_healpix_map = from_cutout_to_fullmap(result_spectral_params, indices, nside)

    # Plot the best and result maps
    plt.figure(figsize=(15, 5))
    hp.mollview(
        best_healpix_map, title="Best Beta Dust map", sub=(1, 2, 1), bgcolor=(0.0,) * 4
    )
    hp.mollview(
        result_healpix_map,
        title="Result Beta Dust map",
        sub=(1, 2, 2),
        bgcolor=(0.0,) * 4,
    )
    plt.savefig(
        f"{out_folder}/best_result_healpix_projection.png",
        dpi=600,
        transparent=True,
    )
