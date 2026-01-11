"""
Test: Active Set vs Adam - Convergence Comparison

Tests that active_set converges faster and better than plain Adam
on standard optimization test functions with minimum inside bounds.
"""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from furax_cs.optim.active_set import active_set

jax.config.update("jax_numpy_dtype_promotion", "standard")


# ============================================================================
# Test Functions
# ============================================================================


def rosenbrock(x):
    """Rosenbrock function. Minimum at [1, 1, ..., 1] with value 0."""
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def himmelblau(x):
    """Himmelblau function. Has 4 minima, all with value 0.
    Minima at: (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
    """
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def sphere(x):
    """Sphere function. Minimum at origin with value 0."""
    return jnp.sum(x**2)


def beale(x):
    """Beale function. Minimum at [3, 0.5] with value 0."""
    return (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )


# ============================================================================
# Optimization Runner
# ============================================================================


def run_optimization(solver, params, value_and_grad_fn, max_steps, lower=None, upper=None):
    """Run optimization, return final params and loss history."""
    state = solver.init(params)
    loss_history = []
    value_fn = lambda p: value_and_grad_fn(p)[0]

    @jax.jit
    def step(params, state):
        val, grad = value_and_grad_fn(params)
        updates, new_state = solver.update(
            grad, state, params, value=val, grad=grad, value_fn=value_fn
        )
        new_params = optax.apply_updates(params, updates)
        if lower is not None:
            new_params = jnp.maximum(new_params, lower)
        if upper is not None:
            new_params = jnp.minimum(new_params, upper)
        return new_params, new_state, val

    for _ in range(max_steps):
        params, state, val = step(params, state)
        jax.block_until_ready(params)
        loss_history.append(float(val))

    return params, loss_history


# ============================================================================
# Test Cases
# ============================================================================


def test_function(name, fn, start, expected_min, bounds_range=100.0, max_steps=500, verbose=False):
    """Test a single function comparing active_set vs baseline (both use Adam + Backtracking)."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"{'=' * 60}")

    dim = len(start)
    params = jnp.array(start)
    lower = -jnp.ones(dim) * bounds_range
    upper = jnp.ones(dim) * bounds_range
    vg = jax.value_and_grad(fn)

    # Both use the same optimizer components
    direction = optax.adam(learning_rate=0.1)
    linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=5)

    # Baseline: Adam + Backtracking chain
    baseline = optax.chain(direction, linesearch)

    # Active Set wrapping Adam + Backtracking
    active = active_set(direction, linesearch, lower=lower, upper=upper, verbose=verbose)

    # Warm-up JIT
    _ = run_optimization(baseline, params, vg, 5, lower, upper)
    _ = run_optimization(active, params, vg, 5, lower, upper)

    # Run Baseline
    t0 = time.perf_counter()
    final_baseline, loss_baseline = run_optimization(baseline, params, vg, max_steps, lower, upper)
    time_baseline = time.perf_counter() - t0

    # Run Active Set
    t0 = time.perf_counter()
    final_active, loss_active = run_optimization(active, params, vg, max_steps, lower, upper)
    time_active = time.perf_counter() - t0

    # Results
    print("\nBaseline (Adam + Backtracking):")
    print(f"  Final loss: {loss_baseline[-1]:.2e}")
    print(f"  Final params: {final_baseline}")
    print(f"  Time: {time_baseline:.3f}s")

    print("\nActive Set:")
    print(f"  Final loss: {loss_active[-1]:.2e}")
    print(f"  Final params: {final_active}")
    print(f"  Time: {time_active:.3f}s")

    # Find iterations to reach threshold
    threshold = 1e-4
    iter_baseline = next((i for i, l in enumerate(loss_baseline) if l < threshold), max_steps)
    iter_active = next((i for i, l in enumerate(loss_active) if l < threshold), max_steps)

    print(f"\nIterations to loss < {threshold}:")
    print(f"  Baseline: {iter_baseline}")
    print(f"  Active Set: {iter_active}")

    # Comparison
    active_better = loss_active[-1] < loss_baseline[-1]
    active_faster = iter_active < iter_baseline
    print(f"\nActive Set better convergence: {'YES' if active_better else 'NO'}")
    print(f"Active Set faster to threshold: {'YES' if active_faster else 'NO'}")

    return {
        "name": name,
        "loss_baseline": loss_baseline,
        "loss_active": loss_active,
        "final_baseline": final_baseline,
        "final_active": final_active,
        "iter_baseline": iter_baseline,
        "iter_active": iter_active,
        "active_better": active_better,
        "active_faster": active_faster,
    }


def plot_results(results):
    """Plot comparison of all test functions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, res in enumerate(results):
        ax = axes[i]
        ax.semilogy(res["loss_baseline"], label="Baseline", linewidth=2, alpha=0.8)
        ax.semilogy(res["loss_active"], label="Active Set", linewidth=2, linestyle="--", alpha=0.8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(res["name"])
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("active_set_comparison.png", dpi=150)
    print("\nSaved: active_set_comparison.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================


VERBOSE = False
MAX_STEPS = 2000

if __name__ == "__main__":
    print("=" * 60)
    print("ACTIVE SET vs ADAM - CONVERGENCE COMPARISON")
    print("=" * 60)

    results = []

    # Test 1: Sphere (10D) - Simple convex
    results.append(
        test_function(
            name="Sphere (10D)",
            fn=sphere,
            start=[3.0] * 10,
            expected_min=[0.0] * 10,
            bounds_range=10.0,
            max_steps=MAX_STEPS,
            verbose=VERBOSE,
        )
    )

    # Test 2: Rosenbrock (2D) - Classic non-convex
    results.append(
        test_function(
            name="Rosenbrock (2D)",
            fn=rosenbrock,
            start=[0.0, 0.0],
            expected_min=[1.0, 1.0],
            bounds_range=5.0,
            max_steps=MAX_STEPS,
            verbose=VERBOSE,
        )
    )

    # Test 3: Himmelblau (2D) - Multiple minima
    results.append(
        test_function(
            name="Himmelblau (2D)",
            fn=himmelblau,
            start=[0.0, 0.0],
            expected_min=[3.0, 2.0],  # One of the minima
            bounds_range=5.0,
            max_steps=MAX_STEPS,
            verbose=VERBOSE,
        )
    )

    # Test 4: Beale (2D) - Flat regions
    results.append(
        test_function(
            name="Beale (2D)",
            fn=beale,
            start=[0.0, 0.0],
            expected_min=[3.0, 0.5],
            bounds_range=5.0,
            max_steps=MAX_STEPS,
            verbose=VERBOSE,
        )
    )

    # Plot
    plot_results(results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Function':<20} {'Active Better?':<15} {'Active Faster?':<15}")
    print("-" * 50)
    for res in results:
        better = "YES" if res["active_better"] else "NO"
        faster = "YES" if res["active_faster"] else "NO"
        print(f"{res['name']:<20} {better:<15} {faster:<15}")

    wins = sum(1 for r in results if r["active_better"])
    print(f"\nActive Set wins: {wins}/{len(results)} functions")
