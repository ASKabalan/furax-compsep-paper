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
