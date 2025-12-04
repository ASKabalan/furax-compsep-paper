"""
Off-the-shelf L-BFGS solvers and function conditioning utilities for optimization.

Solvers:
    - lbfgs_zoom: L-BFGS with zoom linesearch (strong Wolfe conditions)
    - lbfgs_backtrack: L-BFGS with backtracking linesearch (Armijo condition)
    - lbfgs_zoom_safe: L-BFGS zoom with scale_init_precond=False (numerically stable)
    - lbfgs_backtrack_safe: L-BFGS backtracking with scale_init_precond=False

Function Wrappers:
    - precondition: Transform function to work in [0,1] parameter space
    - postcondition: Normalize function output by a factor
    - condition: Apply both precondition and postcondition
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

import jax
import jax.numpy as jnp
import optax
from optax._src import base, combine, transform
from optax._src import linesearch as _linesearch

# =============================================================================
# OFF-THE-SHELF L-BFGS SOLVERS
# =============================================================================


def lbfgs_zoom(
    learning_rate: base.ScalarOrSchedule | None = None,
    memory_size: int = 10,
    scale_init_precond: bool = False,
    max_linesearch_steps: int = 200,
    initial_guess_strategy: str = "one",
    slope_rtol: float = 1e-4,
    curv_rtol: float = 0.9,
    verbose: bool = False,
) -> base.GradientTransformationExtraArgs:
    """L-BFGS with zoom linesearch (strong Wolfe conditions).

    This is the standard L-BFGS with zoom linesearch that enforces both:
    - Sufficient decrease (Armijo): f(x + η*d) ≤ f(x) + c1*η*∇f(x)ᵀd
    - Curvature condition: |∇f(x + η*d)ᵀd| ≤ c2*|∇f(x)ᵀd|

    Args:
        learning_rate: Optional global scaling factor.
        memory_size: Number of past updates for Hessian approximation.
        scale_init_precond: Whether to scale initial Hessian approximation.
            WARNING: Set to False for numerically sensitive problems.
        max_linesearch_steps: Maximum iterations for zoom linesearch.
        initial_guess_strategy: "one" (start at η=1) or "keep" (use previous).
        slope_rtol: c1 parameter for Armijo condition (default 1e-4).
        curv_rtol: c2 parameter for curvature condition (default 0.9).
        verbose: Print linesearch debugging info.

    Returns:
        An optax GradientTransformationExtraArgs.
    """
    if learning_rate is None:
        base_scaling = transform.scale(-1.0)
    else:
        base_scaling = optax.scale_by_learning_rate(learning_rate)

    linesearch = _linesearch.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        initial_guess_strategy=initial_guess_strategy,
        slope_rtol=slope_rtol,
        curv_rtol=curv_rtol,
        verbose=verbose,
    )

    return combine.chain(
        transform.scale_by_lbfgs(
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
        ),
        base_scaling,
        linesearch,
    )


def lbfgs_backtrack(
    learning_rate: base.ScalarOrSchedule | None = None,
    memory_size: int = 10,
    scale_init_precond: bool = False,
    max_backtracking_steps: int = 200,
    slope_rtol: float = 1e-4,
    decrease_factor: float = 0.8,
    increase_factor: float = 1.5,
    max_learning_rate: float = 1.0,
    verbose: bool = False,
) -> base.GradientTransformationExtraArgs:
    """L-BFGS with backtracking linesearch (Armijo condition only).

    Simpler than zoom linesearch, only enforces sufficient decrease:
    - Armijo: f(x + η*d) ≤ f(x) + c1*η*∇f(x)ᵀd

    Args:
        learning_rate: Optional global scaling factor.
        memory_size: Number of past updates for Hessian approximation.
        scale_init_precond: Whether to scale initial Hessian approximation.
            WARNING: Set to False for numerically sensitive problems.
        max_backtracking_steps: Maximum backtracking iterations.
        slope_rtol: c1 parameter for Armijo condition (default 1e-4).
        decrease_factor: Multiply stepsize by this when condition fails (default 0.8).
        increase_factor: Initial guess = previous * this factor (default 1.5).
        max_learning_rate: Upper bound on stepsize (default 1.0).
        verbose: Print linesearch debugging info.

    Returns:
        An optax GradientTransformationExtraArgs.
    """
    if learning_rate is None:
        base_scaling = transform.scale(-1.0)
    else:
        base_scaling = optax.scale_by_learning_rate(learning_rate)

    linesearch = _linesearch.scale_by_backtracking_linesearch(
        max_backtracking_steps=max_backtracking_steps,
        slope_rtol=slope_rtol,
        decrease_factor=decrease_factor,
        increase_factor=increase_factor,
        max_learning_rate=max_learning_rate,
        verbose=verbose,
    )

    return combine.chain(
        transform.scale_by_lbfgs(
            memory_size=memory_size,
            scale_init_precond=scale_init_precond,
        ),
        base_scaling,
        linesearch,
    )


# =============================================================================
# FUNCTION CONDITIONING UTILITIES
# =============================================================================


def condition(
    fn: Callable,
    lower: Any | None = None,
    upper: Any | None = None,
    transform_fn: Any | None = None,
    inv_transform_fn: Any | None = None,
    factor: float = 1.0,
) -> tuple[Callable, Callable, Callable]:
    """Apply parameter transformation and output normalization to a function.

    Supports two modes of parameter transformation:
    1. Min-max scaling: Maps parameters from [lower, upper] to [0, 1]
    2. Custom transforms: Apply arbitrary functions (e.g., log transform)

    Args:
        fn: Function to wrap, fn(params, *args, **kwargs) -> scalar
        lower: Lower bounds for min-max scaling (pytree, same structure as params)
        upper: Upper bounds for min-max scaling (pytree, same structure as params)
        transform_fn: Forward transforms (pytree of callables, e.g., {'T': jnp.log})
                      Applied to convert physical params to optimization space
        inv_transform_fn: Inverse transforms (pytree of callables, e.g., {'T': jnp.exp})
                          Applied to convert optimization params back to physical space
        factor: Normalization factor for output (default 1.0 = no scaling)
                Output = fn(params) / factor

    Returns:
        Tuple of (wrapped_fn, to_opt, from_opt) where:
            - wrapped_fn: Function that takes transformed params
            - to_opt: Convert physical params to optimization space
            - from_opt: Convert optimization params to physical space

    Examples:
        # No conditioning (identity)
        >>> wrapped_fn, to_opt, from_opt = condition(fn)

        # Min-max scaling only
        >>> lower = {'T': 10.0, 'beta': 0.5}
        >>> upper = {'T': 40.0, 'beta': 3.0}
        >>> wrapped_fn, to_opt, from_opt = condition(fn, lower=lower, upper=upper)

        # Custom transforms (log for temperature)
        >>> transform_fn = {
        ...     'temp_dust': jnp.log,
        ...     'beta_dust': lambda x: x,
        ...     'beta_pl': lambda x: x,
        ... }
        >>> inv_transform_fn = {
        ...     'temp_dust': jnp.exp,
        ...     'beta_dust': lambda x: x,
        ...     'beta_pl': lambda x: x,
        ... }
        >>> wrapped_fn, to_opt, from_opt = condition(
        ...     fn, transform_fn=transform_fn, inv_transform_fn=inv_transform_fn
        ... )

        # Output normalization only
        >>> wrapped_fn, to_opt, from_opt = condition(fn, factor=npix * ncomp)

        # Combine min-max with output normalization
        >>> wrapped_fn, to_opt, from_opt = condition(
        ...     fn, lower=lower, upper=upper, factor=npix * ncomp
        ... )
    """
    # Determine which parameter transformation mode to use
    has_bounds = lower is not None and upper is not None
    has_custom = transform_fn is not None and inv_transform_fn is not None

    # Validation
    if has_bounds and has_custom:
        raise ValueError(
            "Cannot specify both (lower, upper) and (transform_fn, inv_transform_fn). "
            "Choose one transformation mode."
        )

    if (lower is None) != (upper is None):
        raise ValueError("Must specify both lower and upper, or neither.")

    if (transform_fn is None) != (inv_transform_fn is None):
        raise ValueError("Must specify both transform_fn and inv_transform_fn, or neither.")

    # Build transformation functions
    if has_bounds:
        # Min-max scaling: physical -> [0, 1]
        def to_opt(params):
            return jax.tree.map(lambda p, lo, hi: (p - lo) / (hi - lo), params, lower, upper)

        def from_opt(opt_params):
            return jax.tree.map(lambda u, lo, hi: u * (hi - lo) + lo, opt_params, lower, upper)

        def clip_opt(opt_params):
            return jax.tree.map(lambda u: jnp.clip(u, 0.0, 1.0), opt_params)

    elif has_custom:
        # Custom transforms: physical -> transformed space
        def to_opt(params):
            return jax.tree.map(lambda p, f: f(p), params, transform_fn)

        def from_opt(opt_params):
            return jax.tree.map(lambda u, f: f(u), opt_params, inv_transform_fn)

        def clip_opt(opt_params):
            # No clipping for custom transforms (user handles bounds if needed)
            return opt_params

    else:
        # Identity transformation
        def to_opt(params):
            return params

        def from_opt(opt_params):
            return opt_params

        def clip_opt(opt_params):
            return opt_params

    # Build wrapped function
    @wraps(fn)
    def wrapped_fn(opt_params, *args, **kwargs):
        physical_params = from_opt(opt_params)
        return fn(physical_params, *args, **kwargs) / factor

    # Attach utilities and metadata
    wrapped_fn.to_opt = to_opt
    wrapped_fn.from_opt = from_opt
    wrapped_fn.clip_opt = clip_opt
    wrapped_fn.factor = factor
    wrapped_fn.original_fn = fn

    # Store transformation info for debugging
    wrapped_fn.mode = "bounds" if has_bounds else ("custom" if has_custom else "identity")
    if has_bounds:
        wrapped_fn.lower = lower
        wrapped_fn.upper = upper
    if has_custom:
        wrapped_fn.transform_fn = transform_fn
        wrapped_fn.inv_transform_fn = inv_transform_fn

    return wrapped_fn, to_opt, from_opt
