"""
Unified optimization interface.

This module contains:
- optimize: Unified optimization interface for optax, optimistix, and scipy solvers
- scipy_minimize: Scipy minimize wrapper with vmap support
- ScipyMinimizeState: State class for scipy_minimize results
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxopt import ScipyBoundedMinimize

from .solvers import SOLVER_NAMES, get_solver
from .utils import condition

# =============================================================================
# SCIPY MINIMIZE WITH VMAP SUPPORT
# =============================================================================


class ScipyMinimizeState(eqx.Module):
    """State returned by scipy minimize via pure_callback.

    This equinox module holds the optimization result in a JAX-compatible format
    that can be used with vmap/lax.map.

    Attributes
    ----------
    params : jax.Array
        Optimized parameters.
    fun_val : jax.Array
        Final objective function value (scalar).
    success : jax.Array
        Whether optimization converged successfully (bool scalar).
    iter_num : jax.Array
        Number of iterations performed (int32 scalar).
    """

    params: jax.Array
    fun_val: jax.Array
    success: jax.Array
    iter_num: jax.Array


def scipy_minimize(
    fn: Callable,
    init_params: jax.Array,
    lower_bound: jax.Array | None = None,
    upper_bound: jax.Array | None = None,
    method: str = "tnc",
    maxiter: int = 1000,
    **fn_kwargs,
) -> ScipyMinimizeState:
    """Scipy minimize wrapper that supports vmap via jax.pure_callback.

    This function wraps scipy optimization in a way that is compatible with
    JAX transformations like vmap and lax.map. It uses jax.pure_callback to
    call the host-side scipy solver.

    Parameters
    ----------
    fn : Callable
        Objective function to minimize. Should accept (params, **fn_kwargs).
    init_params : jax.Array
        Initial parameter values.
    lower_bound : jax.Array, optional
        Lower bounds for parameters. Same shape as init_params.
    upper_bound : jax.Array, optional
        Upper bounds for parameters. Same shape as init_params.
    method : str
        Scipy optimization method (default "tnc").
    maxiter : int
        Maximum number of iterations.
    **fn_kwargs
        Additional arguments passed to fn.

    Returns
    -------
    ScipyMinimizeState
        Optimization result containing params, fun_val, success, and iter_num.

    Examples
    --------
    Single optimization:
    >>> result = scipy_minimize(fn, init_params, lower_bound, upper_bound)

    Batched optimization with lax.map:
    >>> results = jax.lax.map(
    ...     lambda args: scipy_minimize(fn, *args),
    ...     (batched_init, batched_lower, batched_upper)
    ... )
    """

    def host_solver_callback(x_init, lower, upper, fn_kwargs):
        """Host-side scipy solver callback."""
        # Handle bounds
        if lower is None and upper is None:
            bounds = None
        else:
            bounds = (lower, upper)

        # Define wrapped objective
        def scipy_fn(params, fn_kwargs):
            return fn(params, **fn_kwargs)

        solver = ScipyBoundedMinimize(
            fun=scipy_fn,
            method=method,
            jit=False,
            maxiter=maxiter,
            options={"disp": True},
        )

        res = solver.run(x_init, bounds=bounds, fn_kwargs=fn_kwargs)

        # Return numpy arrays for pure_callback
        return {
            "params": jax.tree.map(lambda x: np.array(x), res.params),
            "fun_val": np.array(res.state.fun_val, dtype=np.float32),
            "success": np.array(res.state.success, dtype=bool),
            "iter_num": np.array(res.state.iter_num, dtype=np.int32),
        }

    # Define result shape for pure_callback
    result_shape = {
        "params": jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), init_params),
        "fun_val": jax.ShapeDtypeStruct((), jnp.float32),
        "success": jax.ShapeDtypeStruct((), jnp.bool_),
        "iter_num": jax.ShapeDtypeStruct((), jnp.int32),
    }

    result_dict = jax.pure_callback(
        host_solver_callback,
        result_shape,
        init_params,
        lower_bound,
        upper_bound,
        fn_kwargs,
        vmap_method="sequential",
    )

    return ScipyMinimizeState(
        params=result_dict["params"],
        fun_val=result_dict["fun_val"],
        success=result_dict["success"],
        iter_num=result_dict["iter_num"],
    )


# =============================================================================
# UNIFIED OPTIMIZATION INTERFACE
# =============================================================================


def minimize(
    fn: Callable,
    init_params: Any,
    solver_name: SOLVER_NAMES = "optax_lbfgs_zoom",
    max_iter: int = 1000,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    lower_bound: Any | None = None,
    upper_bound: Any | None = None,
    precondition=False,
    **fn_kwargs,
) -> tuple[Any, Any]:
    """
    Unified optimization interface.

    Supports optax solvers, optimistix solvers (via optimistix.minimise),
    and scipy solvers (via jaxopt.ScipyMinimize).

    Parameters
    ----------
    fn : Callable
        Objective function to minimize. Should accept (params, **fn_kwargs).
    init_params : PyTree
        Initial parameter values.
    solver_name : str
        Solver identifier. See SOLVER_NAMES for available options:
        - optax_lbfgs_zoom, optax_lbfgs_backtrack, adam
        - optimistix_bfgs_armijo, optimistix_bfgs_wolfe
        - optimistix_lbfgs_armijo, optimistix_lbfgs_wolfe
        - optimistix_ncg_{pr,hs,fr,dy}[_wolfe]
        - scipy_tnc
        - zoom, backtrack (legacy aliases)
    max_iter : int
        Maximum iterations.
    rtol, atol : float
        Relative/absolute tolerance for optimization convergence.
    lower_bound, upper_bound : PyTree, optional
        Box constraints for optax solvers (lbfgs_zoom, lbfgs_backtrack, adam).
        Parameters are projected to [lower_bound, upper_bound] after each update.
    progress : ProgressMeter, optional
        Progress meter for tracking (optax/optimistix).
    log_updates : bool
        Whether to log updates (optax only).
    **fn_kwargs
        Additional arguments passed to fn.

    Returns
    -------
    final_params : PyTree
        Optimized parameters.
    final_state : Any
        Final optimizer state.
    """

    if precondition:
        fn, to_opt, from_opt = condition(
            fn,
            lower=lower_bound,
            upper=upper_bound,
            scale_function=precondition,
            init_params=init_params,
            **fn_kwargs,
        )
        init_params = to_opt(init_params)
        lower_bound = to_opt(lower_bound) if lower_bound is not None else None
        upper_bound = to_opt(upper_bound) if upper_bound is not None else None
    else:
        from_opt = lambda x: x

    solver, solver_type = get_solver(
        solver_name, rtol=rtol, atol=atol, lower=lower_bound, upper=upper_bound
    )

    if solver_type == "optimistix":
        # Optimistix uses (y, args) signature, wrap fn
        def optx_fn(y, fn_kwargs):
            return fn(y, **fn_kwargs)

        sol = optx.minimise(
            optx_fn,
            solver,
            init_params,
            max_steps=max_iter,
            progress_meter=optx.TqdmProgressMeter(refresh_steps=10),
            throw=False,
            args=fn_kwargs,
        )
        return from_opt(sol.value), sol

    elif solver_type == "scipy":
        # Scipy via vmap-compatible scipy_minimize
        method = solver_name.split("_")[1]
        state = scipy_minimize(
            fn=fn,
            init_params=init_params,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            method=method,
            maxiter=max_iter,
            **fn_kwargs,
        )
        return from_opt(state.params), state

    else:
        raise ValueError(f"Unknown solver type: {solver_type}")
