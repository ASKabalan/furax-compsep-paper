import os

os.environ["EQX_ON_ERROR"] = "nan"
import os

import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import seaborn as sns
from grid_search_data import select_best_params
from jax_healpy.clustering import get_fullmap_from_cutout
from matplotlib import cm, cycler

# Set the style for the plots
plt.style.use("science")
font_size = 12
plt.rcParams.update(
    {
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "axes.titlesize": font_size,
        "font.family": "serif",  # or 'Times New Roman' to match LaTeX
        "legend.frameon": True,  # Add boxed legends
    }
)
jax.config.update("jax_enable_x64", True)


def filter_constant_param(input_dict, indx):
    return jax.tree.map(lambda x: x[indx], input_dict)


def sort_results(results, key):
    indices = np.argsort(results[key])
    return jax.tree.map(lambda x: x[indices], results)


def plot_cmb_nll_vs_B_d_patches_with_noise_3D(results, best_params, out_folder, nb_to_plot):
    for i, nb in enumerate(nb_to_plot):
        run = filter_constant_param(results, nb)
        B_s = run["B_s_patches"]
        T_d = run["T_d_patches"]

        mask = (results["B_s_patches"] == B_s) & (results["T_d_patches"] == T_d)
        filtered = filter_constant_param(results, mask)
        filtered = sort_results(filtered, "B_d_patches")

        x_vals = filtered["B_d_patches"]  # shape (n_configs,)
        values = filtered["value"]  # shape (n_configs, n_realizations)
        nll = filtered["NLL"]  # same shape

        # Create meshgrid for realization index and B_d
        B_d_grid, realization_grid = np.meshgrid(x_vals, np.arange(values.shape[1]), indexing="ij")

        # Transpose data so dimensions match meshgrid
        variance_grid = values
        nll_grid = -nll  # Negative log-likelihood

        fig = plt.figure(figsize=(10, 8))

        # --- Variance plot ---
        ax1 = fig.add_subplot(211, projection="3d")
        surf1 = ax1.plot_surface(
            B_d_grid, realization_grid, variance_grid, cmap=cm.viridis, edgecolor="none"
        )
        ax1.set_title(
            f"CMB Variance vs $K_{{\\beta_d}}$ and Realization\n($T_d$={T_d}, $B_s$={B_s})"
        )
        ax1.set_xlabel("$K_{\\beta_d}$")
        ax1.set_ylabel("Realization")
        ax1.set_zlabel("CMB Variance")
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

        # --- NLL plot ---
        ax2 = fig.add_subplot(212, projection="3d")
        surf2 = ax2.plot_surface(
            B_d_grid, realization_grid, nll_grid, cmap=cm.inferno, edgecolor="none"
        )
        ax2.set_title(
            f"Negative Log-Likelihood vs $K_{{\\beta_d}}$ and Realization\n($T_d$={T_d}, $B_s$={B_s})"  # noqa: E501
        )
        ax2.set_xlabel("$K_{\\beta_d}$")
        ax2.set_ylabel("Realization")
        ax2.set_zlabel("Negative Log-Likelihood")
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

        plt.tight_layout()
        plt.savefig(
            f"{out_folder}/3d_surface_Td_{T_d}_Bs_{B_s}.pdf",
            dpi=300,
            transparent=False,
        )
        plt.close(fig)


def plot_cmb_nll_vs_B_d_patches_with_noise(results, best_params, out_folder, nb_to_plot, noise_sim_count):
    fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    # Define custom, print-friendly colors
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#4a4a4a"]
    plt.rc("axes", prop_cycle=cycler(color=colors))

    for i, nb in enumerate(nb_to_plot):
        run = filter_constant_param(results, nb)
        B_s = run["B_s_patches"]
        T_d = run["T_d_patches"]

        mask = (results["B_s_patches"] == B_s) & (results["T_d_patches"] == T_d)
        filtered = filter_constant_param(results, mask)
        filtered = sort_results(filtered, "B_d_patches")

        x = filtered["B_d_patches"]
        y_variance_mean = filtered["value"].mean(axis=1)
        y_variance_std = filtered["value"].std(axis=1)
        y_likelihood_mean = -filtered["NLL"].mean(axis=1)
        y_likelihood_std = filtered["NLL"].std(axis=1)

        label = f"$K_{{T_d}}$={int(T_d)}, $K_{{\\beta_s}}$={int(B_s)}"

        # roll one to the right 
        y_variance_mean = np.roll(y_variance_mean, -1)
        y_variance_std = np.roll(y_variance_std, -1)

        # CMB Variance plot: line + scatter + error bars
        axs[0].plot(x, y_variance_mean, "-", alpha=0.7)
        axs[0].errorbar(
            x,
            y_variance_mean,
            yerr=y_variance_std / np.sqrt(noise_sim_count),
            fmt="o",
            label=label,
            capsize=3,
        )

        # Likelihood plot: line + scatter + error bars
        axs[1].plot(x, y_likelihood_mean, "-", alpha=0.7)
        axs[1].errorbar(
            x,
            y_likelihood_mean,
            yerr=y_likelihood_std / np.sqrt(noise_sim_count),
            fmt="o",
            label=label,
            capsize=3,
        )

    # Mark best B_d
    best_B_d = best_params["B_d_patches"]
    axs[0].axvline(best_B_d, color="red", linestyle="--", label=f"Best $K_{{\\beta_d}}$ = {int(best_B_d)}")
    axs[1].axvline(best_B_d, color="red", linestyle="--")

    # Improve axis labels and titles
    axs[0].set_ylabel(r"Mean CMB Variance ($\mu$K²)")
    axs[0].set_title("Mean CMB Variance vs Dust Index Patches ($K_{\\beta_d}$)")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_xlabel("Number of Dust Index Patches ($K_{\\beta_d}$)")
    axs[1].set_ylabel("Mean Likelihood")
    axs[1].set_title("Mean Log-Likelihood vs Dust Index Patches ($K_{\\beta_d}$)")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"{out_folder}/validation_likelihood_vs_variance.pdf",
        dpi=1200,
        transparent=True,
    )


def plot_healpix_projection_with_noise(mask, nside, results, best_params, out_folder, noise_runs):
    sns.set_context("paper")

    (indices,) = jnp.where(mask == 1)

    # Get best run
    best_run = filter_constant_param(results, 0)
    patches = best_run["beta_dust_patches"][0]

    best_spectral_params = best_params["beta_dust"][patches]
    best_healpix_map = get_fullmap_from_cutout(best_spectral_params, indices, nside)

    # Process all runs to compute mean & std deviation
    beta_dust_values = best_run["beta_dust"]

    mean_spectral_params = jnp.mean(beta_dust_values, axis=0)[patches]
    std_spectral_params = jnp.std(beta_dust_values, axis=0)[patches]

    mean_healpix_map = get_fullmap_from_cutout(mean_spectral_params, indices, nside)
    std_dev_map = get_fullmap_from_cutout(std_spectral_params / np.sqrt(noise_runs), indices, nside)

    # Plot results
    plt.figure(figsize=(6, 12))
    hp.mollview(best_healpix_map, title="Best Beta Dust Map", sub=(3, 1, 1), bgcolor=(0.0,) * 4)
    hp.mollview(mean_healpix_map, title="Mean Beta Dust Map", sub=(3, 1, 2), bgcolor=(0.0,) * 4)
    hp.mollview(
        std_dev_map,
        title="Standard Deviation (Uncertainty)",
        sub=(3, 1, 3),
        bgcolor=(0.0,) * 4,
    )

    plt.tight_layout()
    plt.savefig(f"{out_folder}/beta_dust_projection.pdf", dpi=1200, transparent=True)


def plot_cmb_nll_vs_B_d_patches(results, best_params, out_folder):
    sns.set_context("paper")
    # Extract values from the grid search results
    B_d_patches = results["B_d_patches"]  # dust patch count (x-axis)
    cmb_variance = results["value"]  # CMB variance values from grid search
    nll = results["NLL"]  # Negative log-likelihood values

    # Create subplots: one for CMB variance, one for NLL
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Handle case where values have multiple realizations (shape: n_configs, n_realizations)
    if cmb_variance.ndim > 1 and cmb_variance.shape[1] > 1:
        # Compute mean and std across noise realizations
        variance_mean = np.mean(cmb_variance, axis=1)
        variance_std = np.std(cmb_variance, axis=1)
        nll_mean = np.mean(nll, axis=1)
        nll_std = np.std(nll, axis=1)
        
        # Plot with error bars
        axs[0].errorbar(B_d_patches, variance_mean, yerr=variance_std, 
                        fmt='o-', capsize=5, capthick=2, color="blue", label="Grid Search")
        axs[1].errorbar(B_d_patches, nll_mean, yerr=nll_std,
                        fmt='o-', capsize=5, capthick=2, color="green", label="Grid Search")
    else:
        # Single realization case - use scatter plot
        axs[0].scatter(B_d_patches, cmb_variance, color="blue", label="Grid Search")
        axs[1].scatter(B_d_patches, nll, color="green", label="Grid Search")

    # Add best parameter lines
    axs[0].axhline(y=best_params["value"], color="red", linestyle="--", label="Best CMB Variance")
    axs[0].axvline(
        x=best_params["B_d_patches"],
        color="orange",
        linestyle="--",
        label=r"Best $K_{\beta_d}$",
    )
    axs[0].set_xlabel(r"$K_{\beta_d}$")  # Updated to K-notation
    axs[0].set_ylabel(r"CMB Variance ($\mu$K²)")  # Added units
    axs[0].set_title(r"CMB Variance vs. $K_{\beta_d}$")  # Updated title
    axs[0].legend()

    # Plot NLL vs. B_d_patches
    axs[1].axhline(y=best_params["NLL"], color="red", linestyle="--", label="Best NLL")
    axs[1].axvline(
        x=best_params["B_d_patches"],
        color="orange",
        linestyle="--",
        label=r"Best $K_{\beta_d}$",
    )
    axs[1].set_xlabel(r"$K_{\beta_d}$")  # Updated to K-notation
    axs[1].set_ylabel("Negative Log Likelihood")
    axs[1].set_title(r"NLL vs. $K_{\beta_d}$")  # Updated title
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(
        f"{out_folder}/cmb_nll_vs_B_d_patches.png",
        dpi=1200,
        transparent=True,
    )


def plot_healpix_projection(mask, nside, results, best_params, out_folder):
    sns.set_context("paper")

    (indices,) = jnp.where(mask == 1)
    best_run = filter_constant_param(results, 0)
    patches = patches = best_run["beta_dust_patches"]
    best_spectral_params = best_params["beta_dust"][patches]
    result_spectral_params = best_run["beta_dust"][patches]

    best_healpix_map = get_fullmap_from_cutout(best_spectral_params, indices, nside)
    result_healpix_map = get_fullmap_from_cutout(result_spectral_params, indices, nside)

    # Plot the best and result maps
    plt.figure(figsize=(15, 5))
    hp.mollview(best_healpix_map, title="Best Beta Dust map", sub=(1, 2, 1), bgcolor=(0.0,) * 4)
    hp.mollview(
        result_healpix_map,
        title="Result Beta Dust map",
        sub=(1, 2, 2),
        bgcolor=(0.0,) * 4,
    )
    plt.savefig(
        f"{out_folder}/best_result_healpix_projection.png",
        dpi=1200,
        transparent=True,
    )


def plot_grid_search_results(
    results, out_folder, best_metric="value", nb_best=9, plot_style="errorbars", noise_runs=50
):
    """
    1) Select the nb_best combos by the chosen best_metric ('value' or 'NLL').
    2) Plot them in a grid (3 columns x enough rows), each cell split into two panels:
       - Top: 'value' vs. B_d
       - Bottom: 'NLL' vs. B_d
    3) Optionally plot with error bars or just a line (plot_style).

    results: dict with keys:
       ['T_d_patches','B_s_patches','B_d_patches','value','NLL']
       where 'value' and 'NLL' are lists of arrays (#B_d_points, #noise_runs).

    best_metric: which metric to use for selecting combos: 'value' or 'NLL'.
    nb_best: number of combos to display.
    plot_style: 'errorbars' or 'line'.
    """
    sns.set_context("paper")

    # 1) Select combos
    combos, combos_best_value, combos_best_nll = select_best_params(
        results, best_metric=best_metric, nb_best=nb_best
    )

    # We'll define the global best among the selected combos for reference lines
    global_best_value = combos_best_value.min()
    global_best_nll = combos_best_nll.min()

    # 2) Prepare data arrays
    T_d_arr = np.array(results["T_d_patches"])
    B_s_arr = np.array(results["B_s_patches"])
    B_d_arr = np.array(results["B_d_patches"])

    # 3) Setup figure with a 3-column layout
    nb_cols = 3
    nb_rows = int(np.ceil(len(combos) / nb_cols))
    fig = plt.figure(figsize=(6 * nb_cols, 6 * nb_rows))

    outer_gs = gridspec.GridSpec(
        nrows=nb_rows,
        ncols=nb_cols,
        wspace=0.1,  # less horizontal space between columns
        hspace=0.15,  # less vertical space between rows
        left=0.05,  # left margin
        right=0.98,  # right margin
        top=0.95,  # top margin
        bottom=0.05,  # bottom margin
    )

    for i, (T_d, B_s) in enumerate(combos):
        row_idx = i // nb_cols
        col_idx = i % nb_cols

        # Create sub-GridSpec for top/bottom within each outer cell
        cell_gs = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer_gs[row_idx, col_idx],
            height_ratios=[3, 2],
            hspace=0.1,
            wspace=0.1,
        )
        ax_top = fig.add_subplot(cell_gs[0])
        ax_bottom = fig.add_subplot(cell_gs[1], sharex=ax_top)

        # Filter data for this combo
        indices = np.where((T_d_arr == T_d) & (B_s_arr == B_s))[0]
        value_list = results["value"][indices]  # shape (#B_d_points, #noise_runs)
        nll_list = results["NLL"][indices]
        B_d_vals = B_d_arr[indices]

        # Compute means/stds
        mean_value = np.array([v.mean() for v in value_list])
        std_value = np.array([v.std() for v in value_list]) / np.sqrt(noise_runs)
        mean_nll = np.array([n.mean() for n in nll_list])
        std_nll = np.array([n.std() for n in nll_list]) / np.sqrt(noise_runs)

        # Sort by B_d for a cleaner left-to-right plot
        sort_idx = np.argsort(B_d_vals)
        B_d_vals = B_d_vals[sort_idx]
        mean_value = mean_value[sort_idx]
        std_value = std_value[sort_idx]
        mean_nll = mean_nll[sort_idx]
        std_nll = std_nll[sort_idx]

        # ---- Top panel: "value" vs B_d ----
        if plot_style == "errorbars":
            ax_top.errorbar(
                B_d_vals, mean_value, yerr=std_value, fmt="o", color="blue", label="value"
            )
        else:
            ax_top.plot(B_d_vals, mean_value, "o-", color="blue", label="value")

        # Reference line: global best among the selected combos
        ax_top.axhline(global_best_value, color="r", linestyle="--", label="global best value")
        ax_top.set_ylabel("Mean Value")
        ax_top.set_title(f"T_d={T_d}, B_s={B_s}")
        ax_top.legend(loc="best")

        # ---- Bottom panel: "NLL" vs B_d ----
        if plot_style == "errorbars":
            ax_bottom.errorbar(
                B_d_vals, mean_nll, yerr=std_nll, fmt="o", color="green", label="NLL"
            )
        else:
            ax_bottom.plot(B_d_vals, mean_nll, "o-", color="green", label="NLL")

        ax_bottom.axhline(global_best_nll, color="r", linestyle="--", label="global best NLL")
        ax_bottom.set_xlabel("B_d")
        ax_bottom.set_ylabel("Mean NLL")
        ax_bottom.legend(loc="best")

        # Hide x-labels in top panel to avoid repetition
        plt.setp(ax_top.get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.savefig(
        f"{out_folder}/grid_search_results.png",
        dpi=1200,
        transparent=True,
    )
