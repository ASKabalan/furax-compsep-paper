import re

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .logging_utils import success, warning

plt.style.use("science")

out_folder = "plots/"
font_size = 14
plt.rcParams.update(
    {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.titlesize": font_size,
        "font.family": "serif",
        "legend.frameon": True,
    }
)

_OUTPUT_FORMAT = "png"


def get_run_color(index):
    """Get consistent color for run based on position.

    Position-based color assignment:
    - 1st run: red
    - 2nd run: blue
    - 3rd run: green
    - 4th+ runs: tab10 colormap colors

    Parameters
    ----------
    index : int
        Zero-based index of the run

    Returns
    -------
    color
        Color specification (string or RGB tuple from colormap)
    """
    base_colors = ["red", "blue", "green"]
    base_colors = ["red", "green", "purple"]
    if index < len(base_colors):
        return base_colors[index]
    else:
        return plt.cm.tab10(index % 10)


def _truncate_name_if_too_long(name, max_length=50):
    """Truncate long names for plot titles and filenames."""
    if len(name) > max_length:
        return name[: max_length - 3] + "..."
    return name


def set_output_format(output_format):
    """Set global output format for all plotting functions."""
    global _OUTPUT_FORMAT
    _OUTPUT_FORMAT = output_format


def set_font_size(size):
    """Set global font size for all plotting functions.

    Parameters
    ----------
    size : int
        Font size to use for all plot elements.
    """
    global font_size
    font_size = size
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "axes.titlesize": font_size,
        }
    )


def save_or_show(filename):
    """Save figure to file or show inline based on output format."""
    if _OUTPUT_FORMAT == "show":
        plt.show()
    else:
        ext = "pdf" if _OUTPUT_FORMAT == "pdf" else "png"
        dpi = 300 if ext == "png" else None
        filename = _truncate_name_if_too_long(filename)
        filepath = f"{out_folder}{filename}.{ext}"
        plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        success(f"Saved: {filepath}")


def get_min_variance(cmb_map):
    """Select the realization with minimum variance across Q/U components."""
    seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_map)
    cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_map, seen_mask)
    variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
    variance = sum(jax.tree.leaves(variance))
    argmin = jnp.argmin(variance)
    return jax.tree.map(lambda x: x[argmin], cmb_map)


def plot_params(name, params, plot_vertical=False):
    """Plot recovered spectral parameter maps for a single configuration."""
    with plt.rc_context(
        {
            "font.size": font_size * 1.6,
            "axes.labelsize": font_size * 1.6,
            "xtick.labelsize": font_size * 1.6,
            "ytick.labelsize": font_size * 1.6,
            "legend.fontsize": font_size * 1.6,
            "axes.titlesize": font_size * 1.6,
        }
    ):
        if plot_vertical:
            fig_size = (8, 16)
            subplot_args = (3, 1, lambda i: i + 1)
        else:
            fig_size = (16, 8)
            subplot_args = (1, 3, lambda i: i + 1)

        _ = plt.figure(figsize=fig_size)

        keys = ["beta_dust", "temp_dust", "beta_pl"]
        names = ["$\\beta_d$", "$T_d$", "$\\beta_s$"]

        for i, (key, param_name) in enumerate(zip(keys, names)):
            param_map = params[key]
            hp.mollview(
                param_map,
                title=f"{name} {param_name}",
                sub=(subplot_args[0], subplot_args[1], subplot_args[2](i)),
                bgcolor=(0.0,) * 4,
                cbar=True,
            )

        save_or_show(f"params_{name}")
        params_dict = {
            "beta_dust": params["beta_dust"],
            "temp_dust": params["temp_dust"],
            "beta_pl": params["beta_pl"],
        }
        np.savez(f"{out_folder}/params_{name}.npz", **params_dict)


def plot_patches(name, patches, plot_vertical=False):
    """Visualise patch assignments (cluster labels) for each spectral parameter."""
    with plt.rc_context(
        {
            "font.size": font_size * 1.6,
            "axes.labelsize": font_size * 1.6,
            "xtick.labelsize": font_size * 1.6,
            "ytick.labelsize": font_size * 1.6,
            "legend.fontsize": font_size * 1.6,
            "axes.titlesize": font_size * 1.6,
        }
    ):
        if plot_vertical:
            fig_size = (8, 16)
            subplot_args = (3, 1, lambda i: i + 1)
        else:
            fig_size = (16, 8)
            subplot_args = (1, 3, lambda i: i + 1)

        _ = plt.figure(figsize=fig_size)

        np.random.seed(0)

        def shuffle_labels(arr):
            unique_vals = np.unique(arr[arr != hp.UNSEEN])
            shuffled_vals = np.random.permutation(unique_vals)

            mapping = dict(zip(unique_vals, shuffled_vals))

            shuffled_arr = np.vectorize(lambda x: mapping.get(x, hp.UNSEEN))(arr)
            return shuffled_arr.astype(np.float64)

        patches_dict = {
            "beta_dust_patches": patches["beta_dust_patches"],
            "temp_dust_patches": patches["temp_dust_patches"],
            "beta_pl_patches": patches["beta_pl_patches"],
        }
        np.savez(f"{out_folder}/patches_{name}.npz", **patches_dict)
        patches = jax.tree.map(shuffle_labels, patches)

        keys = ["beta_dust_patches", "temp_dust_patches", "beta_pl_patches"]
        names = ["$\\beta_d$ Patches", "$T_d$ Patches", "$\\beta_s$ Patches"]

        for i, (key, patch_name) in enumerate(zip(keys, names)):
            patch_map = patches[key]
            hp.mollview(
                patch_map,
                title=f"{name} {patch_name}",
                sub=(subplot_args[0], subplot_args[1], subplot_args[2](i)),
                bgcolor=(0.0,) * 4,
                cbar=True,
            )
        save_or_show(f"patches_{name}")


def plot_validation_curves(name, updates_history, value_history):
    """Plot optimizer update norms and NLL traces for each run."""
    updates_history = np.array(updates_history)
    value_history = np.array(value_history)

    n_runs = updates_history.shape[0]
    ncols = 2
    nrows = int(np.ceil(n_runs))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))

    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(n_runs):
        updates = updates_history[i].mean(axis=0)
        values = value_history[i].mean(axis=0)

        valid_mask = values != 0.0
        updates = updates[: len(valid_mask)]
        values = values[valid_mask]
        updates = updates[valid_mask]
        indx = np.arange(len(values))

        axs[i, 0].plot(indx, updates, label=f"Run {i + 1} Updates")
        axs[i, 0].set_title(f"{name} - Updates History Run {i + 1}")
        axs[i, 0].set_xlabel("Iteration")
        axs[i, 0].set_ylabel("Update Norm")
        axs[i, 0].grid(True)
        axs[i, 0].legend()

        axs[i, 1].plot(indx, values, label=f"Run {i + 1} NLL")
        axs[i, 1].set_title(f"{name} - NLL History Run {i + 1}")
        axs[i, 1].set_xlabel("Iteration")
        axs[i, 1].set_ylabel("Negative Log-Likelihood")
        axs[i, 1].grid(True)
        axs[i, 1].legend()

    plt.tight_layout()
    save_or_show(f"validation_curves_{name}")


def plot_all_cmb(names, cmb_pytree_list):
    """Show reconstructed-minus-true Q/U differences for multiple runs."""
    nb_cmb = len(cmb_pytree_list)

    diff_all = []

    for cmb_pytree in cmb_pytree_list:
        cmb_recon = get_min_variance(cmb_pytree["cmb_recon"])

        diff_q = cmb_pytree["cmb"].q - cmb_recon.q
        diff_u = cmb_pytree["cmb"].u - cmb_recon.u

        unseen_mask_q = cmb_pytree["cmb"].q == hp.UNSEEN
        unseen_mask_u = cmb_pytree["cmb"].u == hp.UNSEEN

        diff_q = np.where(unseen_mask_q, np.nan, diff_q)
        diff_u = np.where(unseen_mask_u, np.nan, diff_u)

        diff_all.append((diff_q, diff_u))

    plt.figure(figsize=(10, 3.5 * nb_cmb))

    for i, (name, (diff_q, diff_u)) in enumerate(zip(names, diff_all)):
        hp.mollview(
            diff_q,
            title=rf"Difference (Q) - {name} ($\mu$K)",
            sub=(nb_cmb, 2, 2 * i + 1),
            cbar=True,
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            notext=True,
        )
        hp.mollview(
            diff_u,
            title=rf"Difference (U) - {name} ($\mu$K)",
            sub=(nb_cmb, 2, 2 * i + 2),
            cbar=True,
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            notext=True,
        )

    name = "_".join(names)
    save_or_show(f"cmb_recon_{name}")


def plot_all_variances(names, cmb_pytree_list):
    """Histogram proxy metrics (variance, NLL, ∑Cℓ) across runs."""

    def get_all_variances(cmb_map):
        seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_map)
        cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_map, seen_mask)
        variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
        variance = sum(jax.tree.leaves(variance))
        return variance

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=False)

    metrics = {
        "Variance of Reconstructed CMB (Q + U)": [],
        "Negative Log-Likelihood": [],
        r"$\sum C_\ell^{BB}$": [],
    }

    for name, cmb_pytree in zip(names, cmb_pytree_list):
        metrics["Variance of Reconstructed CMB (Q + U)"].append(
            (name, get_all_variances(cmb_pytree["cmb_recon"]))
        )
        metrics["Negative Log-Likelihood"].append((name, np.array(cmb_pytree["nll_summed"])))
        metrics[r"$\sum C_\ell^{BB}$"].append((name, np.array(cmb_pytree["cl_bb_sum"])))

    for ax, (title, entries) in zip(axs, metrics.items()):
        for i, (name, values) in enumerate(entries):
            color = get_run_color(i)
            label = f"{name}"
            ax.hist(
                values,
                bins=20,
                alpha=0.5,
                label=label,
                color=color,
                edgecolor="black",
                histtype="stepfilled",
            )
            mean_val = np.mean(values)
            ax.axvline(mean_val, color=color, linestyle="--", linewidth=2, label=f"Mean of {name}")

        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Count", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize="small", loc="best")

        for label in ax.get_xticklabels():
            label.set_rotation(30)

    axs[-1].set_xlabel("Metric Value", fontsize=12)

    plt.tight_layout(pad=2.0)
    name = "_".join(names)
    name = "all_metrics"
    save_or_show(f"metric_distributions_histogram_{name}")


def plot_all_cl_residuals(names, cl_pytree_list):
    """Overlay residual BB spectra for all requested configurations."""
    _ = plt.figure(figsize=(8, 6))

    if len(cl_pytree_list) == 0:
        warning("No power spectra results to plot")
        return

    cl_bb_r1 = cl_pytree_list[0]["cl_bb_r1"]
    ell_range = cl_pytree_list[0]["ell_range"]
    cl_bb_lens = cl_pytree_list[0]["cl_bb_lens"]

    r_lo, r_hi = 1e-3, 4e-3
    plt.fill_between(
        ell_range,
        r_lo * cl_bb_r1,
        r_hi * cl_bb_r1,
        color="grey",
        alpha=0.35,
        label=r"$C_\ell^{BB},\; r\in[10^{-3},\,4\cdot10^{-3}]$",
    )

    plt.plot(
        ell_range,
        cl_bb_lens,
        label=r"$C_\ell^{BB}\,\mathrm{lens}$",
        color="grey",
        linestyle="-",
        linewidth=2,
    )

    for i, (name, cl_pytree) in enumerate(zip(names, cl_pytree_list)):
        color = get_run_color(i)
        linewidth = 1.5

        if cl_pytree["cl_total_res"] is not None:
            plt.plot(
                ell_range,
                cl_pytree["cl_total_res"],
                label=rf"{name} $C_\ell^{{\mathrm{{res}}}}$",
                color=color,
                linestyle="--",
            )
        if cl_pytree["cl_syst_res"] is not None:
            plt.plot(
                ell_range,
                cl_pytree["cl_syst_res"],
                label=rf"{name} $C_\ell^{{\mathrm{{syst}}}}$",
                color=color,
                linestyle="-",
                linewidth=linewidth,
            )
        if cl_pytree["cl_stat_res"] is not None:
            plt.plot(
                ell_range,
                cl_pytree["cl_stat_res"],
                label=rf"{name} $C_\ell^{{\mathrm{{stat}}}}$",
                color=color,
                linestyle=":",
                linewidth=linewidth,
            )

    plt.title(None)
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$C_\ell^{BB}$ [$\mu K^2$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    name = "_".join(names)
    save_or_show(f"bb_spectra_{name}")


def plot_all_systematic_residuals(names, syst_map_list):
    """Plot systematic residual Q/U maps for multiple runs."""
    nb_runs = len(syst_map_list)
    if nb_runs == 0:
        warning("No systematic residual maps available to plot")
        return

    plt.figure(figsize=(12, 4 * nb_runs))

    for i, (name, syst_map) in enumerate(zip(names, syst_map_list)):
        syst_q = np.where(syst_map[1] == hp.UNSEEN, np.nan, syst_map[1])
        syst_u = np.where(syst_map[2] == hp.UNSEEN, np.nan, syst_map[2])

        hp.mollview(
            syst_q,
            title=rf"Systematic Residual (Q) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 1),
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

        hp.mollview(
            syst_u,
            title=rf"Systematic Residual (U) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 2),
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

    name = "_".join(names)
    save_or_show(f"all_systematic_residuals_{name}")


def plot_all_statistical_residuals(names, stat_map_list):
    """Plot statistical residual Q/U maps for multiple runs."""
    nb_runs = len(stat_map_list)
    if nb_runs == 0:
        warning("No statistical residual maps available to plot")
        return

    plt.figure(figsize=(12, 4 * nb_runs))

    for i, (name, stat_maps) in enumerate(zip(names, stat_map_list)):
        stat_map_first = stat_maps[0]

        stat_q = np.where(stat_map_first[1] == hp.UNSEEN, np.nan, stat_map_first[1])
        stat_u = np.where(stat_map_first[2] == hp.UNSEEN, np.nan, stat_map_first[2])

        hp.mollview(
            stat_q,
            title=rf"Statistical Residual (Q) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 1),
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

        hp.mollview(
            stat_u,
            title=rf"Statistical Residual (U) - {name} ($\mu$K)",
            sub=(nb_runs, 2, 2 * i + 2),
            min=-0.5,
            max=0.5,
            cmap="RdBu_r",
            bgcolor=(0,) * 4,
            cbar=True,
            notext=True,
        )

    name = "_".join(names)
    save_or_show(f"all_statistical_residuals_{name}")


def plot_all_r_estimation(names, r_pytree_list):
    """Compare r likelihood curves across runs in a single figure."""
    plt.figure(figsize=(8, 6))

    for i, (name, r_data) in enumerate(zip(names, r_pytree_list)):
        if r_data["r_best"] is None:
            warning(f"No r estimation for {name}, skipping plot.")
            continue

        r_grid = r_data["r_grid"]
        L_vals = r_data["L_vals"]
        r_best = r_data["r_best"]
        sigma_r_neg = r_data["sigma_r_neg"]
        sigma_r_pos = r_data["sigma_r_pos"]

        color = get_run_color(i)
        likelihood = L_vals / L_vals.max()

        plt.plot(
            r_grid,
            likelihood,
            label=rf"{name} $\hat{{r}} = {r_best:.2e}^{{+{sigma_r_pos:.1e}}}_{{-{sigma_r_neg:.1e}}}$",
            color=color,
        )

        plt.fill_between(
            r_grid,
            0,
            likelihood,
            where=(r_grid > r_best - sigma_r_neg) & (r_grid < r_best + sigma_r_pos),
            color=color,
            alpha=0.2,
        )

        plt.axvline(
            x=r_best,
            color=color,
            linestyle="--",
            alpha=0.7,
        )

    plt.axvline(x=0.0, color="black", linestyle="--", alpha=0.7, label="True r=0")

    plt.title("Likelihood Curves for $r$ (All Runs)")
    plt.xlabel(r"$r$")
    plt.ylabel("Relative Likelihood")
    plt.grid(True, which="both", ls=":")
    plt.legend(fontsize="medium")
    plt.tight_layout()
    name = "_".join(names)
    save_or_show(f"r_likelihood_{name}")


def _create_r_vs_clusters_plot(patch_name, patch_key, names, cmb_pytree_list, r_pytree_list):
    """Scatter plot of r + σ(r) vs clusters for one parameter."""
    method_dict = {}
    base_patch_keys = ["beta_dust_patches", "temp_dust_patches", "beta_pl_patches"]

    for name, cmb_pytree, r_data in zip(names, cmb_pytree_list, r_pytree_list):
        if r_data["r_best"] is None:
            warning(f"No r estimation for {name}, skipping plot.")
            continue

        base_name = re.sub(r" \(\d+\)$", "", name)

        patches = cmb_pytree["patches_map"]
        if patch_key == "total":
            n_clusters = 0
            for key in base_patch_keys:
                patch_data = patches[key]
                n_clusters += np.unique(patch_data[patch_data != hp.UNSEEN]).size
        else:
            patch_data = patches[patch_key]
            n_clusters = np.unique(patch_data[patch_data != hp.UNSEEN]).size

        if n_clusters in method_dict:
            existing_r_values = method_dict[n_clusters]["r_best"]
            if r_data["r_best"] > existing_r_values:
                continue

        method_dict[n_clusters] = {
            "name": base_name,
            "r_best": r_data["r_best"],
            "sigma_r_neg": r_data["sigma_r_neg"],
            "sigma_r_pos": r_data["sigma_r_pos"],
        }

    plt.figure(figsize=(8, 6))

    if len(method_dict) == 0:
        warning(f"No valid data points for {patch_key} in r_vs_clusters plot.")
        plt.close()
        return

    sorted_items = sorted(method_dict.items(), key=lambda item: item[0])

    cluster_points = []
    r_plus_sigma_vals = []
    min_point = None

    for idx, (n_clusters, data) in enumerate(sorted_items):
        r_best = data["r_best"]
        sigma_r_pos = data["sigma_r_pos"]
        r_plus_sigma = r_best + sigma_r_pos

        cluster_points.append(n_clusters)
        r_plus_sigma_vals.append(r_plus_sigma)

        if (min_point is None) or (r_plus_sigma < min_point["value"]):
            min_point = {
                "index": idx,
                "clusters": n_clusters,
                "value": r_plus_sigma,
            }

    scatter_all = plt.scatter(
        cluster_points,
        r_plus_sigma_vals,
        color="#1f77b4",
        s=100,
        edgecolors="black",
        linewidths=1,
    )

    if min_point is not None:
        min_idx = min_point["index"]
        min_clusters = cluster_points[min_idx]
        min_value = r_plus_sigma_vals[min_idx]

        scatter_min = plt.scatter(
            [min_clusters],
            [min_value],
            color="red",
            s=120,
            edgecolors="black",
            linewidths=1.5,
            zorder=3,
        )
        min_label = (
            r"Lowest $r+\sigma(r)$ at "
            f"{int(min_clusters)} clusters: {min_value:.2e}"
        )
    else:
        scatter_min = None
        min_label = None

    legend_handles = [scatter_all]
    legend_labels = [r"$r+\sigma(r)$"]
    if scatter_min is not None and min_label is not None:
        legend_handles.append(scatter_min)
        legend_labels.append(min_label)

    plt.xlabel(f"Number of Clusters ({patch_name})")
    plt.ylabel(r"$r + \sigma(r)$")
    plt.title(r"$r + \sigma(r)$ vs. Number of Clusters" + f" ({patch_name})")
    # plt.ylim(-0.001, 0.01)
    plt.axhline(y=0.0, color="black", linestyle="--", alpha=0.7, linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.6)

    if legend_handles:
        plt.legend(legend_handles, legend_labels)

    plt.tight_layout()

    filename_suffix = patch_key.replace("_patches", "")
    save_or_show(f"r_vs_clusters_{filename_suffix}")
    plt.close()


def _create_variance_vs_clusters_plot(patch_name, patch_key, names, cmb_pytree_list):
    """Scatter plot of cluster count vs minimum variance for one parameter."""
    method_dict = {}
    base_patch_keys = ["beta_dust_patches", "temp_dust_patches", "beta_pl_patches"]
    other_patch_keys = [k for k in base_patch_keys if k != patch_key]

    for name, cmb_pytree in zip(names, cmb_pytree_list):
        base_name = re.sub(r" \(\d+\)$", "", name)

        patches = cmb_pytree["patches_map"]
        if patch_key == "total":
            n_clusters = 0
            for key in base_patch_keys:
                patch_data = patches[key]
                n_clusters += np.unique(patch_data[patch_data != hp.UNSEEN]).size
            total_clusters = n_clusters
        else:
            patch_data = patches[patch_key]
            n_clusters = np.unique(patch_data[patch_data != hp.UNSEEN]).size
            total_clusters = n_clusters
            for other_key in other_patch_keys:
                other_patch_data = patches[other_key]
                total_clusters += np.unique(other_patch_data[other_patch_data != hp.UNSEEN]).size

        seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_pytree["cmb_recon"])
        cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_pytree["cmb_recon"], seen_mask)
        variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
        variance = sum(jax.tree.leaves(variance))
        min_variance = float(jnp.min(variance))

        if n_clusters in method_dict:
            existing_variance = method_dict[n_clusters]["variance"]
            if min_variance > existing_variance:
                continue

        method_dict[n_clusters] = {
            "name": base_name,
            "variance": min_variance,
            "total_clusters": total_clusters,
        }

    plt.figure(figsize=(8, 6))

    if len(method_dict) == 0:
        warning(f"No valid data points for {patch_key} in variance_vs_clusters plot.")
        plt.close()
        return

    sorted_items = sorted(method_dict.items(), key=lambda item: item[0])
    total_cluster_values = np.array([data["total_clusters"] for _, data in sorted_items])
    total_min = float(total_cluster_values.min())
    total_max = float(total_cluster_values.max())
    if total_min == total_max:
        total_min -= 0.5
        total_max += 0.5

    cmap = plt.cm.viridis
    norm = Normalize(vmin=total_min, vmax=total_max)

    for (n_clusters, data), total_clusters in zip(sorted_items, total_cluster_values):
        variance = data["variance"]
        color = cmap(norm(total_clusters))

        plt.scatter(
            n_clusters,
            variance,
            color=color,
            s=100,
            edgecolors="black",
            linewidths=1,
        )

    plt.xlabel(f"Number of Clusters ({patch_name})")
    plt.ylabel(r"Minimum Variance (Q + U)")
    plt.title(f"Minimum Variance vs. Number of Clusters ({patch_name})")
    plt.grid(True, linestyle="--", alpha=0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label("Total Number of Clusters")

    plt.tight_layout()

    filename_suffix = patch_key.replace("_patches", "")
    save_or_show(f"variance_vs_clusters_{filename_suffix}")
    plt.close()


def plot_variance_vs_clusters(names, cmb_pytree_list):
    """Plot minimum recovered-CMB variance vs cluster count for each parameter."""
    patch_configs = [
        ("$\\beta_d$", "beta_dust_patches"),
        ("$T_d$", "temp_dust_patches"),
        ("$\\beta_s$", "beta_pl_patches"),
        ("Total", "total"),
    ]

    for patch_name, patch_key in patch_configs:
        _create_variance_vs_clusters_plot(patch_name, patch_key, names, cmb_pytree_list)


def _create_variance_vs_r_plot(patch_name, patch_key, names, cmb_pytree_list, r_pytree_list):
    """Helper to plot variance vs best-fit r for a given parameter/totals."""
    points = []
    is_total = patch_key == "total"

    for name, cmb_pytree, r_data in zip(names, cmb_pytree_list, r_pytree_list):
        if r_data["r_best"] is None:
            warning(f"No r estimation for {name}, skipping variance_vs_r point.")
            continue

        patches = cmb_pytree["patches_map"]

        if is_total:
            n_clusters = 0
            for key in ["beta_dust_patches", "temp_dust_patches", "beta_pl_patches"]:
                patch_data = patches[key]
                n_clusters += np.unique(patch_data[patch_data != hp.UNSEEN]).size
        else:
            patch_data = patches[patch_key]
            n_clusters = np.unique(patch_data[patch_data != hp.UNSEEN]).size

        seen_mask = jax.tree.map(lambda x: jnp.all(x != hp.UNSEEN, axis=0), cmb_pytree["cmb_recon"])
        cmb_map_seen = jax.tree.map(lambda x, m: x[:, m], cmb_pytree["cmb_recon"], seen_mask)
        variance = jax.tree.map(lambda x: jnp.var(x, axis=1), cmb_map_seen)
        variance = sum(jax.tree.leaves(variance))
        min_variance = float(jnp.min(variance))

        points.append(
            (
                min_variance,
                float(r_data["r_best"]),
                int(n_clusters),
                float(r_data["sigma_r_neg"]),
                float(r_data["sigma_r_pos"]),
            )
        )

    if len(points) == 0:
        warning("No valid data points for variance_vs_r plot.")
        return

    points.sort(key=lambda p: p[0])
    variances = [p[0] for p in points]
    r_values = [p[1] for p in points]
    k_values = np.array([p[2] for p in points])
    sigma_r_neg = [p[3] for i, p in enumerate(points)]
    sigma_r_pos = [p[4] for i, p in enumerate(points)]

    plt.figure(figsize=(8, 6))

    cmap = plt.cm.viridis
    norm = Normalize(vmin=k_values.min(), vmax=k_values.max())
    colors = cmap(norm(k_values))

    for i in range(len(variances)):
        plt.errorbar(
            variances[i],
            r_values[i],
            yerr=[[sigma_r_neg[i]], [sigma_r_pos[i]]],
            fmt="o",
            color=colors[i],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=8,
            capsize=3,
            elinewidth=1.5,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    if is_total:
        cbar.set_label("Total Number of Clusters")
    else:
        cbar.set_label(f"Number of Clusters ({patch_name})")

    plt.xlabel(r"Minimum Variance (Q + U)")
    plt.ylabel(r"Best-fit $r$")
    plt.ylim(-0.0005, 0.005)
    plt.title(f"Variance vs $r$ ({patch_name})")
    plt.axhline(y=0.0, color="black", linestyle="--", alpha=0.7, linewidth=1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    filename_suffix = "total" if is_total else patch_key.replace("_patches", "")
    save_or_show(f"variance_vs_r_{filename_suffix}")
    plt.close()


def plot_variance_vs_r(names, cmb_pytree_list, r_pytree_list):
    """Plot variance vs best-fit r for each spectral parameter and combined."""
    patch_configs = [
        ("$\\beta_d$", "beta_dust_patches"),
        ("$T_d$", "temp_dust_patches"),
        ("$\\beta_s$", "beta_pl_patches"),
        ("Total", "total"),
    ]

    for patch_name, patch_key in patch_configs:
        _create_variance_vs_r_plot(patch_name, patch_key, names, cmb_pytree_list, r_pytree_list)


def plot_r_vs_clusters(names, cmb_pytree_list, r_pytree_list):
    """Plot r + σ(r) vs number of clusters for each parameter."""
    patch_configs = [
        ("$\\beta_d$", "beta_dust_patches"),
        ("$T_d$", "temp_dust_patches"),
        ("$\\beta_s$", "beta_pl_patches"),
        ("Total", "total"),
    ]

    for patch_name, patch_key in patch_configs:
        _create_r_vs_clusters_plot(patch_name, patch_key, names, cmb_pytree_list, r_pytree_list)


def plot_systematic_residual_maps(name, syst_map):
    """Plot systematic residual Q/U maps for a single configuration."""
    syst_q = np.where(syst_map[1] == hp.UNSEEN, np.nan, syst_map[1])
    syst_u = np.where(syst_map[2] == hp.UNSEEN, np.nan, syst_map[2])

    plt.figure(figsize=(12, 6))

    hp.mollview(
        syst_q,
        title=rf"Systematic Residual (Q) - {name} ($\mu$K)",
        sub=(1, 2, 1),
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    hp.mollview(
        syst_u,
        title=rf"Systematic Residual (U) - {name} ($\mu$K)",
        sub=(1, 2, 2),
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    save_or_show(f"systematic_residual_maps_{name}")


def plot_statistical_residual_maps(name, stat_maps):
    """Plot statistical residual Q/U maps for a single configuration."""
    stat_map_first = stat_maps[0]

    stat_q = np.where(stat_map_first[1] == hp.UNSEEN, np.nan, stat_map_first[1])
    stat_u = np.where(stat_map_first[2] == hp.UNSEEN, np.nan, stat_map_first[2])

    plt.figure(figsize=(12, 6))

    hp.mollview(
        stat_q,
        title=rf"Statistical Residual (Q) - {name} ($\mu$K)",
        sub=(1, 2, 1),
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    hp.mollview(
        stat_u,
        title=rf"Statistical Residual (U) - {name} ($\mu$K)",
        sub=(1, 2, 2),
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
        cbar=True,
        notext=True,
    )

    save_or_show(f"statistical_residual_maps_{name}")


def plot_cmb_reconstructions(name, cmb_stokes, cmb_recon):
    """Plot reconstructed maps, inputs, and differences for Q/U."""

    def mse(a, b):
        seen_x = jax.tree.map(lambda x: x[x != hp.UNSEEN], a)
        seen_y = jax.tree.map(lambda x: x[x != hp.UNSEEN], b)
        return jax.tree.map(lambda x, y: jnp.mean((x - y) ** 2), seen_x, seen_y)

    cmb_recon_min = get_min_variance(cmb_recon)
    unseen_mask = cmb_recon_min.q == hp.UNSEEN
    diff_q = cmb_recon_min.q - cmb_stokes.q
    diff_q = np.where(unseen_mask, hp.UNSEEN, diff_q)

    unseen_mask = cmb_recon_min.u == hp.UNSEEN
    diff_u = cmb_recon_min.u - cmb_stokes.u
    diff_u = np.where(unseen_mask, hp.UNSEEN, diff_u)

    _ = plt.figure(figsize=(12, 12))
    hp.mollview(
        cmb_recon_min.q, title=r"Reconstructed CMB (Q) [$\mu$K]", sub=(3, 3, 1), bgcolor=(0,) * 4
    )
    hp.mollview(cmb_stokes.q, title=r"Input CMB Map (Q) [$\mu$K]", sub=(3, 3, 2), bgcolor=(0,) * 4)
    hp.mollview(
        diff_q,
        title=r"Difference (Q) [$\mu$K]",
        sub=(3, 3, 3),
        cbar=True,
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
    )
    hp.mollview(
        cmb_recon_min.u, title=r"Reconstructed CMB (U) [$\mu$K]", sub=(3, 3, 4), bgcolor=(0,) * 4
    )
    hp.mollview(cmb_stokes.u, title=r"Input CMB Map (U) [$\mu$K]", sub=(3, 3, 5), bgcolor=(0,) * 4)
    hp.mollview(
        diff_u,
        title=r"Difference (U) [$\mu$K]",
        sub=(3, 3, 6),
        cbar=True,
        min=-0.5,
        max=0.5,
        cmap="RdBu_r",
        bgcolor=(0,) * 4,
    )
    plt.title(f"{name} CMB Reconstruction")
    save_or_show(f"cmb_recon_{name}")


def plot_cl_residuals(
    name,
    cl_bb_obs,
    cl_syst_res,
    cl_total_res,
    cl_stat_res,
    cl_bb_r1,
    cl_bb_r0,
    cl_bb_lens,
    cl_true,
    ell_range,
):
    """Plot detailed BB spectrum decomposition for a single configuration."""
    _ = plt.figure(figsize=(12, 8))

    coeff = ell_range * (ell_range + 1) / (2 * np.pi)

    plt.plot(ell_range, cl_bb_obs * coeff, label=r"$C_\ell^{\mathrm{obs}}$", color="green")
    plt.plot(ell_range, cl_total_res * coeff, label=r"$C_\ell^{\mathrm{res}}$", color="black")
    plt.plot(ell_range, cl_syst_res * coeff, label=r"$C_\ell^{\mathrm{syst}}$", color="blue")
    plt.plot(ell_range, cl_stat_res * coeff, label=r"$C_\ell^{\mathrm{stat}}$", color="orange")
    plt.plot(ell_range, cl_bb_r1 * coeff, label=r"$C_\ell^{\mathrm{BB}}(r=1)$", color="red")
    plt.plot(ell_range, cl_bb_r0 * coeff, label=r"$C_\ell^{\mathrm{BB}}(r=0)$", color="orange")
    plt.plot(
        ell_range,
        cl_true * coeff,
        label=r"$C_\ell^{\mathrm{true}}$",
        color="purple",
        linestyle="--",
    )
    plt.plot(
        ell_range,
        cl_bb_lens * coeff,
        label=r"$C_\ell^{\mathrm{lens}}$",
        color="purple",
        linestyle=":",
    )

    plt.title(f"{name} BB Power Spectra")
    plt.xlabel(r"Multipole $\ell$")
    plt.ylabel(r"$D_\ell^{BB}$ [$\mu K^2$]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    save_or_show(f"bb_spectra_{name}")


def plot_r_estimator(
    name,
    r_best,
    sigma_r_neg,
    sigma_r_pos,
    r_grid,
    L_vals,
):
    """Plot one-dimensional likelihood for r with highlighted estimate."""
    plt.figure(figsize=(12, 8))

    likelihood = L_vals / np.max(L_vals)

    plt.plot(
        r_grid,
        likelihood,
        label=rf"{name} $\hat{{r}} = {r_best:.2e}^{{+{sigma_r_pos:.1e}}}_{{-{sigma_r_neg:.1e}}}$",
        color="purple",
        linewidth=2,
    )
    plt.fill_between(
        r_grid,
        0,
        likelihood,
        where=(r_grid > r_best - sigma_r_neg) & (r_grid < r_best + sigma_r_pos),
        color="purple",
        alpha=0.2,
    )
    plt.axvline(r_best, color="purple", linestyle="--", alpha=0.8)

    plt.axvline(0.0, color="purple", linestyle="--", alpha=0.8, label="True r=0")
    plt.title(f"{name} Likelihood vs $r$")
    plt.xlabel(r"$r$")
    plt.ylabel("Relative Likelihood")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_or_show(f"r_likelihood_{name}")

    print(f"Estimated r (Reconstructed): {r_best:.4e} (+{sigma_r_pos:.1e}, -{sigma_r_neg:.1e})")
