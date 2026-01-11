"""
Function conditioning utilities.

This module contains the condition function for:
- Parameter transformation (min-max scaling to [0, 1])
- Gradient-based output scaling (like scipy TNC's fscale)
"""

from collections.abc import Callable
from functools import wraps
from typing import Any

import jax
from jaxtyping import PyTree
from lineax.internal import two_norm


def condition(
    fn: Callable[..., Any],
    lower: Any | None = None,
    upper: Any | None = None,
    scale_function: bool = False,
    init_params: Any | None = None,
    *args,
    **kwargs,
) -> tuple[Callable[..., Any], Callable[[PyTree], PyTree], Callable[[PyTree], PyTree]]:
    """Apply parameter transformation and gradient-based output scaling.

    Transforms parameters to [0, 1] space using min-max scaling and optionally
    scales the function output based on gradient norm at initial parameters
    (like scipy TNC's fscale).

    Args:
        fn: Function to wrap, fn(params, *args, **kwargs) -> scalar
        lower: Lower bounds for min-max scaling (pytree, same structure as params)
        upper: Upper bounds for min-max scaling (pytree, same structure as params)
        scale_function: If True, compute fscale from gradient norm at init_params
        init_params: Initial parameters for computing fscale (required if scale_function=True)
        *args, **kwargs: Additional arguments passed to fn for gradient computation

    Returns:
        Tuple of (wrapped_fn, to_opt, from_opt) where:
            - wrapped_fn: Function that takes transformed params and returns scaled output
            - to_opt: Convert physical params to optimization space [0, 1]
            - from_opt: Convert optimization params back to physical space

    Example:
        >>> fn, to_opt, from_opt = condition(
        ...     negative_log_likelihood,
        ...     lower={'beta': 0.5, 'temp': 10.0},
        ...     upper={'beta': 3.0, 'temp': 40.0},
        ...     scale_function=True,
        ...     init_params={'beta': 1.5, 'temp': 20.0},
        ...     # kwargs for fn:
        ...     nu=frequencies,
        ...     d=data,
        ... )
        >>> opt_params = to_opt(init_params)  # transform to [0, 1]
        >>> result = fn(opt_params, nu=frequencies, d=data)  # scaled output
        >>> physical_params = from_opt(opt_params)  # back to physical
    """
    has_bounds = lower is not None and upper is not None

    # Build transformation functions
    if has_bounds:
        # Min-max scaling: physical -> [0, 1]
        def to_opt(params: PyTree) -> PyTree:
            return jax.tree.map(lambda p, lo, hi: (p - lo) / (hi - lo), params, lower, upper)

        def from_opt(opt_params: PyTree) -> PyTree:
            return jax.tree.map(lambda u, lo, hi: u * (hi - lo) + lo, opt_params, lower, upper)
    else:
        # Identity transformation
        def to_opt(params: PyTree) -> PyTree:
            return params

        def from_opt(opt_params: PyTree) -> PyTree:
            return opt_params

    # Compute fscale from gradient if requested
    factor = 1.0
    if scale_function:
        if init_params is None:
            raise ValueError("init_params required when scale_function=True")

        # Build unscaled wrapped function
        def wrapped_fn_unscaled(opt_params: PyTree, *a: Any, **kw: Any) -> Any:
            physical_params = from_opt(opt_params)
            return fn(physical_params, *a, **kw)

        # Compute gradient at init_params
        opt_init_params = to_opt(init_params)
        grad = jax.grad(wrapped_fn_unscaled)(opt_init_params, *args, **kwargs)
        gnorm = two_norm(grad)
        factor = 1.0 / gnorm

    # Build final wrapped function
    @wraps(fn)
    def wrapped_fn(opt_params: PyTree, *a: Any, **kw: Any) -> Any:
        physical_params = from_opt(opt_params)
        return fn(physical_params, *a, **kw) * factor

    # Attach utilities and metadata
    wrapped_fn.to_opt = to_opt  # type: ignore[attr-defined]
    wrapped_fn.from_opt = from_opt  # type: ignore[attr-defined]
    wrapped_fn.factor = factor  # type: ignore[attr-defined]
    wrapped_fn.original_fn = fn  # type: ignore[attr-defined]

    return wrapped_fn, to_opt, from_opt
