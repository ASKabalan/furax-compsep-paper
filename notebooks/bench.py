# Necessary imports
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

# Healpy and PySM3 imports

# FGBuster imports
from fgbuster import (
    CMB,
    Dust,
    Synchrotron,
    MixingMatrix,
    get_observation,
    get_instrument,
)
from fgbuster.algebra import _build_bound_inv_logL_and_logL_dB
from fgbuster import (
    basic_comp_sep,
)

# Furax imports
import jax
import jaxopt
import jax.numpy as jnp
from furax._base.blocks import BlockDiagonalOperator, BlockRowOperator
from furax._base.core import HomothetyOperator, IdentityOperator
from furax.landscapes import StokesPyTree, HealpixLandscape
from furax.operators.seds import CMBOperator, DustOperator, SynchrotronOperator
import operator
import optax
from furax.comp_sep import optimize
from generate_maps import save_to_cache
from jax_hpc_profiler import Timer
from furax.comp_sep import spectral_cmb_variance, negative_log_likelihood
from functools import partial
from generate_maps import load_from_cache
import seaborn as sns
import matplotlib.pyplot as plt
from jax_hpc_profiler.plotting import plot_weak_scaling , plot_strong_scaling



def run_fgbuster_logL(nside, freq_maps, components, nu , numpy_timer):
    """Run FGBuster log-likelihood."""
    print(f'Running FGBuster Log Likelihood with nside={nside} ...')

    A = MixingMatrix(*components)
    A_ev = A.evaluator(nu)
    A_dB_ev = A.diff_evaluator(nu)
    data = freq_maps.T

    logL, _, _ = _build_bound_inv_logL_and_logL_dB(A_ev, data, None, A_dB_ev, A.comp_of_dB)
    x0 = np.array([x for c in components for x in c.defaults])

    durations = []
    numpy_timer.chrono_jit(logL , x0)
    for _ in range(10):
        numpy_timer.chrono_fun(logL , x0)



def run_jax_negative_log_prob(
    nside, freq_maps, best_params, nu, dust_nu0, synchrotron_nu0 , jax_timer
):
    """Run JAX-based negative log-likelihood."""
    print(f'Running Furax Log Likelihood nside={nside} ...')
    d = StokesPyTree.from_stokes(I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])
    invN = HomothetyOperator(jnp.ones(1), _in_structure=d.structure)

    nll = partial(negative_log_likelihood , nu=nu, N=invN, d=d , dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0 )


    jax_timer.chrono_jit(nll , best_params)

    for _ in range(10):
        jax_timer.chrono_fun(nll , best_params)

     
def run_jax_lbfgs(nside, freq_maps, best_params, nu, dust_nu0, synchrotron_nu0 , jax_timer):
    """Run JAX-based negative log-likelihood."""

    print(f'Running Furax LBGS Comp sep nside={nside} ...')

    d = StokesPyTree.from_stokes(I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])
    invN = HomothetyOperator(jnp.ones(1), _in_structure=d.structure)

    nll = partial(negative_log_likelihood , nu=nu, N=invN, d=d , dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0 )

    solver = optax.lbfgs()

    def basic_comp(best_params):
        final_params, _ = optimize(
            best_params,
            nll,
            solver,
            max_iter=100,
            tol=1e-5,
        )
        return final_params['beta_pl'] , final_params

    jax_timer.chrono_jit(basic_comp , best_params , ndarray_arg=0)
    for _ in range(10):
        jax_timer.chrono_fun(basic_comp , best_params, ndarray_arg=0)


def run_jax_tnc(nside, freq_maps, best_params, nu, dust_nu0, synchrotron_nu0, numpy_timer):
    """Run JAX-based negative log-likelihood."""

    print(f'Running Furax TNC From SciPy Comp sep nside={nside} ...')

    d = StokesPyTree.from_stokes(I=freq_maps[:, 0, :], Q=freq_maps[:, 1, :], U=freq_maps[:, 2, :])
    invN = HomothetyOperator(jnp.ones(1), _in_structure=d.structure)

    nll = partial(negative_log_likelihood , nu=nu, N=invN, d=d , dust_nu0=dust_nu0, synchrotron_nu0=synchrotron_nu0 )

    def basic_comp(best_params):
        scipy_solver = jaxopt.ScipyMinimize(fun=nll, method='TNC', jit=True, tol=1e-6)
        result = scipy_solver.run(best_params)
        return result.params

    numpy_timer.chrono_jit(basic_comp , best_params)
    for _ in range(10):
        numpy_timer.chrono_fun(basic_comp , best_params)


def run_fgbuster_comp_sep(nside, instrument, best_params, freq_maps, components , numpy_timer):
    """Run FGBuster log-likelihood."""
    print(f'Running FGBuster Comp sep nside={nside} ...')

    components[1]._set_default_of_free_symbols(
        beta_d=best_params['beta_dust'], temp=best_params['temp_dust']
    )
    components[2]._set_default_of_free_symbols(beta_pl=best_params['beta_pl'])

    numpy_timer.chrono_jit(basic_comp_sep , components, instrument, freq_maps)

    for _ in range(10):
        numpy_timer.chrono_fun(basic_comp_sep , components, instrument, freq_maps)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark FGBuster and Furax Component Separation Methods'
    )
    parser.add_argument(
        '-n',
        '--nsides',
        type=int,
        nargs='+',
        default=[32, 64, 128, 256, 512],
        help='List of nsides to benchmark',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='comparison',
        help='Output filename prefix for the plots',
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=(10, 6),
        help='Figure size for the plots (width, height)',
    )
    parser.add_argument(
        '-l',
        '--likelihood',
        action='store_true',
        help='Benchmark FGBuster and Furax log-likelihood methods',
    )
    parser.add_argument(
        '-s',
        '--solvers',
        action='store_true',
        help='Benchmark solvers: FGBuster, JAX LBFGS, and JAX TNC',
    )
    parser.add_argument(
        '-p',
        '--plot-only',
        action='store_true',
        help='Benchmark solvers: FGBuster, JAX LBFGS, and JAX TNC',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    instrument = get_instrument('LiteBIRD')
    nu = instrument['frequency'].values
    stokes_type = 'IQU'
    dust_nu0, synchrotron_nu0 = 150.0, 20.0
    components = [CMB(), Dust(dust_nu0), Synchrotron(synchrotron_nu0)]
    best_params = {'temp_dust': 20.0, 'beta_dust': 1.54, 'beta_pl': -3.0}


    jax_timer = Timer(save_jaxpr=True, jax_fn=True)
    numpy_timer = Timer(save_jaxpr=False, jax_fn=False)

    if args.likelihood and not args.plot_only:
        for nside in args.nsides:
            save_to_cache(nside, sky='c1d0s0')
            nu , freq_maps = load_from_cache(nside , sky='c1d0s0')

            # Likelihood mode benchmarking
            print(f'Running likelihood benchmarking for nside={nside}...')
            run_fgbuster_logL(nside, freq_maps, components, nu , numpy_timer)
            run_jax_negative_log_prob(nside, freq_maps, best_params, nu, dust_nu0, synchrotron_nu0 , jax_timer)

            kwargs = {"function": "Furax", "precision": "float64", "x": nside}
            jax_timer.report("runs/FURAX.csv", **kwargs)
            kwargs = {"function": "FGBuster", "precision": "float64", "x": nside}
            numpy_timer.report("runs/FGBUSTER.csv", **kwargs)

    if args.solvers and not args.plot_only:
        for nside in args.nsides:
            save_to_cache(nside, sky='c1d0s0')
            nu , freq_maps = load_from_cache(nside , sky='c1d0s0')

            # Solver mode benchmarking
            print(f'Running solver benchmarking for nside={nside}...')
            run_fgbuster_comp_sep(
                nside, instrument, best_params, freq_maps, components, numpy_timer
            )
            kwargs = {"function": "FGBuster - TNC", "precision": "float64", "x": nside}
            numpy_timer.report("runs/FGBUSTER.csv", **kwargs)

            # Run JAX LBFGS from Optax
            #run_jax_lbfgs(
            #    nside, freq_maps, best_params, nu, dust_nu0, synchrotron_nu0 , jax_timer
            #)
            #kwargs = {"function": "Furax - LBFGS", "precision": "float64", "x": nside}
            #jax_timer.report("runs/FURAX.csv", **kwargs)

            # Run TNC from SciPy
            run_jax_tnc(
                nside, freq_maps, best_params, nu, dust_nu0, synchrotron_nu0 , numpy_timer
            )
            kwargs = {"function": "Furax - TNC", "precision": "float64", "x": nside}
            numpy_timer.report("runs/FURAX.csv", **kwargs)




    # Plot log-likelihood results
    if args.likelihood:

        plt.rcParams.update({'font.size': 15})
        sns.set_context("paper")


        csv_file = ["runs/FURAX.csv" , "runs/FGBUSTER.csv"]
        FWs = ["Furax" , "FGBuster"]

        plot_weak_scaling(
            csv_files=csv_file,
            functions=FWs,
            figure_size=(12, 8),
            label_text="%f%",
            output="runs/nll.png"
        )

    # Plot solver results
    if args.solvers:
        plt.rcParams.update({'font.size': 15})
        sns.set_context("paper")


        csv_file = ["runs/FURAX.csv" , "runs/FGBUSTER.csv"]
        solvers = ["Furax - TNC" , "Furax - LBFGS" , "FGBuster - TNC"]

        plot_weak_scaling(
            csv_files=csv_file,
            functions=solvers,
            figure_size=(12, 8),
            label_text="%f%",
            output="runs/solvers.png"
        )


if __name__ == '__main__':
    main()
