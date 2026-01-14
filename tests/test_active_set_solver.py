import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from furax_cs.optim.active_set import active_set


def test_rosenbrock_constrained():
    print("\n--- Running Rosenbrock Test (Constrained) ---")

    def rosenbrock(p):
        a = p["a"]
        b = p["b"]
        return jnp.sum((1.0 - a) ** 2 + 100.0 * (b - a**2) ** 2)

    # Standard Rosenbrock min is (1,1).
    # We constrain a <= 0.5. The optimizer should hit the wall at a=0.5.

    params = {"a": jnp.array([0.0]), "b": jnp.array([0.0])}
    lower = {"a": jnp.array([-jnp.inf]), "b": jnp.array([-jnp.inf])}
    upper = {"a": jnp.array([0.5]), "b": jnp.array([jnp.inf])}  # Block access to 1.0

    # Solvers
    # Use Adam for direction
    direction = optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8)
    # Use simple backtracking. Note: optax's backtracking applies 'updates' to 'params'
    linesearch = optax.scale_by_backtracking_linesearch(max_backtracking_steps=15, store_grad=True)

    solver = active_set(direction, linesearch, lower=lower, upper=upper, verbose=True)
    state = solver.init(params)

    val_grad = jax.value_and_grad(rosenbrock)

    print(f"Start: {params}")

    for i in range(2):
        val, grad = val_grad(params)
        updates, state = solver.update(
            grad, state, params, value=val, grad=grad, value_fn=rosenbrock
        )
        params = optax.apply_updates(params, updates)
        # Numerical stability clip
        params = jtu.tree_map(jnp.clip, params, lower, upper)

        if i % 100 == 0:
            print(
                f"Iter {i}: val={val:.5f} a={params['a'][0]:.5f} b={params['b'][0]:.5f} piv_a={state.pivot['a'][0]}"
            )

    print(f"Final: a={params['a'][0]:.5f} b={params['b'][0]:.5f}")
    print(f"Final Pivot: a={state.pivot['a'][0]}")

    print(params)
    # Check convergence
    assert jnp.allclose(params["a"], 0.5, atol=1e-3)
    # With a=0.5, b should minimize 100(b - 0.25)^2 -> b=0.25
    assert jnp.allclose(params["b"], 0.25, atol=1e-2)
    assert state.pivot["a"][0] == 1  # Active upper bound
    print("Test Passed!")


if __name__ == "__main__":
    test_rosenbrock_constrained()
