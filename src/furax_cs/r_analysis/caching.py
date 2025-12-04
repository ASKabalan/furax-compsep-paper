import hashlib
import json
import os
import pickle
import re
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from furax import HomothetyOperator
from furax.obs import negative_log_likelihood, sky_signal
from furax.obs.stokes import Stokes
from jax_grid_search import ProgressBar, optimize
from rich.progress import BarColumn, TimeElapsedColumn, TimeRemainingColumn
from tqdm import tqdm

from .logging_utils import success, warning


def compute_w(nu, d, results, result_file, run_index=0, max_iter=100, force_recompute=False):
    """Compute or load the foreground-only CMB reconstruction (W·d_fg).

    Parameters
    ----------
    nu : array_like
        Frequency array in GHz.
    d : Stokes
        Foreground-only frequency maps.
    results : dict
        Results dictionary potentially containing cached W_D_FG.
    result_file : str
        Path to results.npz file for saving cache.
    run_index : int, optional
        Noise realization index (default: 0).
    max_iter : int, optional
        Maximum L-BFGS iterations (default: 100).
    force_recompute : bool, optional
        Force recomputation even if cache exists (default: False).

    Returns
    -------
    Stokes
        Foreground-only CMB reconstruction (W·d_fg).
    """
    cache_key = f"W_D_FG_{run_index}"
    if results.get(cache_key) is not None and not force_recompute:
        print(f"Using {cache_key} from results")
        W = results[cache_key]
        W = Stokes.from_stokes(Q=W[0], U=W[1])
        return W

    dust_nu0 = 160.0
    synchrotron_nu0 = 20.0

    patches = {k: results[k] for k in ["beta_dust_patches", "beta_pl_patches", "temp_dust_patches"]}
    max_count = {
        "beta_dust": patches["beta_dust_patches"].size,
        "temp_dust": patches["temp_dust_patches"].size,
        "beta_pl": patches["beta_pl_patches"].size,
    }

    base_params = {
        "beta_dust": 1.54,
        "temp_dust": 20.0,
        "beta_pl": -3.0,
    }

    guess_params = jax.tree.map(lambda v, c: jnp.full((c,), v), base_params, max_count)

    N = HomothetyOperator(1.0, _in_structure=d.structure)

    negative_log_likelihood_fn = partial(
        negative_log_likelihood, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0
    )

    progress_columns = [
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]

    with ProgressBar(*progress_columns) as p:
        solver = optax.lbfgs()
        final_params, final_state = optimize(
            guess_params,
            negative_log_likelihood_fn,
            solver,
            max_iter=max_iter,
            tol=1e-10,
            progress=p,
            progress_id=0,
            nu=nu,
            N=N,
            d=d,
            patch_indices=patches,
        )

    def W(p):
        N = HomothetyOperator(1.0, _in_structure=d.structure)
        return sky_signal(
            p, nu, N, d, dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0, patch_indices=patches
        )["cmb"]

    W = W(final_params)

    results_from_file = dict(np.load(result_file))
    W_numpy = np.stack([W.q, W.u], axis=0)
    results_from_file[cache_key] = W_numpy
    atomic_save_results(result_file, results_from_file)
    return W


def atomic_save_results(result_file, results_dict):
    """Atomically write updated results to disk with backup protection."""
    if os.path.exists(result_file):
        backup_file = result_file.replace(".npz", ".bk.npz")
        os.rename(result_file, backup_file)

    temp_file = result_file.replace(".npz", ".tmp.npz")
    np.savez(temp_file, **results_dict)

    os.rename(temp_file, result_file)


def check_cache_keys_exist(result_file, run_index):
    """Return True if a cached W_D_FG entry exists for the run index."""
    try:
        with np.load(result_file) as f:
            w_key = f"W_D_FG_{run_index}"
            return w_key in f.keys()
    except (OSError, FileNotFoundError):
        return False


def load_run_data_for_cache(folder, nside, instrument, run_index=0):
    """Load minimal inputs required to compute cached W_D_FG products."""
    run_data = dict(np.load(f"{folder}/results.npz"))
    best_params = dict(np.load(f"{folder}/best_params.npz"))
    mask = np.load(f"{folder}/mask.npy")
    (indices,) = jnp.where(mask == 1)
    f_sky = mask.sum() / len(mask)

    first_key = next(iter(run_data.keys()))
    max_index = len(run_data[first_key]) - 1

    if run_index > max_index:
        warning(
            f"Index {run_index} out of bounds (max: {max_index}) for folder {folder}. Skipping."
        )
        return None

    from .utils import index_run_data

    run_data = index_run_data(run_data, run_index)

    fg_map = Stokes.from_stokes(Q=best_params["I_D_NOCMB"][:, 0], U=best_params["I_D_NOCMB"][:, 1])
    cmb_recon = Stokes.from_stokes(Q=run_data["CMB_O"][:, 0], U=run_data["CMB_O"][:, 1])

    return run_data, best_params, mask, indices, f_sky, cmb_recon, fg_map


def cache_expensive_computations(
    name, filtered_results, nside, instrument, run_index=0, force_recompute=False, max_iter=100
):
    """Materialise expensive W_D_FG caches for a group of runs.

    Parameters
    ----------
    name : str
        Identifier for this run group.
    filtered_results : list of str
        List of result folder paths.
    nside : int
        HEALPix resolution parameter.
    instrument : FGBusterInstrument
        Instrument configuration.
    run_index : int, optional
        Noise realization index (default: 0).
    force_recompute : bool, optional
        Force recomputation even if cache exists (default: False).
    """
    if len(filtered_results) == 0:
        warning(f"No results found matching filter criteria for '{name}'")
        return

    for folder in tqdm(filtered_results, desc=f"Caching {name}", unit="folder"):
        cache_exists = check_cache_keys_exist(f"{folder}/results.npz", run_index)

        if cache_exists and not force_recompute:
            print(f"    Cache already exists for run {name} index {run_index}, skipping...")
            continue

        if cache_exists and force_recompute:
            print(f"    Force recomputing cache for run {name} index {run_index}...")

        try:
            result = load_run_data_for_cache(folder, nside, instrument, run_index)
            if result is None:
                continue

            run_data, best_params, mask, indices, f_sky, cmb_recon, fg_map = result

            if not cache_exists:
                print("    Computing/caching W_D_FG...")
            _ = compute_w(
                instrument.frequency,
                fg_map,
                run_data,
                result_file=f"{folder}/results.npz",
                run_index=run_index,
                force_recompute=force_recompute,
                max_iter=max_iter,
            )

            success(f"Completed {folder}")

        except Exception as e:
            warning(f"Error processing folder {folder}: {e}")
            continue


SNAPSHOT_MANIFEST_NAME = "manifest.json"
SNAPSHOT_VERSION = 1


def _snapshot_filename_from_title(title):
    """Generate a stable filename slug for a snapshot entry."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    if not slug:
        slug = "entry"
    return f"{slug}_{digest}.pkl"


def _tree_to_numpy(tree):
    """Convert JAX arrays to numpy arrays recursively."""

    def _convert_leaf(x):
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "__array__"):
            return np.asarray(x)
        return x

    return jax.tree.map(_convert_leaf, tree)


def _tree_to_jax(tree):
    """Convert numpy arrays to JAX arrays recursively."""

    def _convert_leaf(x):
        if isinstance(x, np.ndarray):
            return jnp.asarray(x)
        return x

    return jax.tree.map(_convert_leaf, tree)


def load_snapshot(snapshot_dir):
    """Load cached analysis payloads from disk snapshots."""
    snapshot_path = Path(snapshot_dir)
    manifest_path = snapshot_path / SNAPSHOT_MANIFEST_NAME

    if not manifest_path.exists():
        return [], {"version": SNAPSHOT_VERSION, "entries": []}

    with manifest_path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    entries = []
    for item in manifest.get("entries", []):
        title = item.get("title")
        filename = item.get("file")
        if title is None or filename is None:
            continue
        payload_path = snapshot_path / filename
        if not payload_path.exists():
            print(f"WARNING: Snapshot payload missing for '{title}' at {payload_path}")
            continue
        with payload_path.open("rb") as fh:
            payload = pickle.load(fh)
        if isinstance(payload, dict):
            converted_payload = {}
            for key in ("cmb", "cl", "r", "residual", "plotting_data"):
                value = payload.get(key)
                if value is not None:
                    converted_payload[key] = _tree_to_jax(value)
                else:
                    converted_payload[key] = value
            payload = converted_payload
        entries.append((title, payload))

    return entries, manifest


def save_snapshot_entry(snapshot_dir, manifest, title, payload):
    """Persist a single snapshot payload and update the manifest."""
    snapshot_path = Path(snapshot_dir)
    snapshot_path.mkdir(parents=True, exist_ok=True)

    entries = manifest.setdefault("entries", [])
    lookup = {item["title"]: item for item in entries if "title" in item}

    existing_entry = lookup.get(title)
    filename = None
    if existing_entry is not None:
        filename = existing_entry.get("file")
    if not filename:
        filename = _snapshot_filename_from_title(title)

    payload_path = snapshot_path / filename
    with payload_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    if existing_entry is not None:
        existing_entry["file"] = filename
    else:
        entries.append({"title": title, "file": filename})

    manifest["version"] = SNAPSHOT_VERSION
    manifest["entries"] = entries
    return manifest


def write_snapshot_manifest(snapshot_dir, manifest):
    """Write the manifest JSON for stored snapshot entries."""
    snapshot_path = Path(snapshot_dir)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    manifest_path = snapshot_path / SNAPSHOT_MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
