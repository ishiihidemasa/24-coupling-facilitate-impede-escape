from typing import Callable

import numpy as np


class Result(dict):
    """
    Represents the simulation result.
    Intended to behave in a similar manner to `scipy.optimize.OptimizeResult`,
    which is the base class of `res` returned by `scipy.integrate.solve_ivp()`.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__

    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys())


def euler_maruyama(
    drift: Callable,
    drift_args: tuple,
    diffusion: Callable,
    diff_args: tuple,
    t_calc: np.ndarray,
    y0: np.ndarray,
    t_eval: np.ndarray = None,
    rng: np.random.Generator = None,
    seed: int = None,
):
    """
    Parameters
    ----------
    drift : Callable
        Drift coefficient in the SDE returning `float` or `np.ndarray`.
        Current time `t`, state variables `y_old` and other args are passed as `drift(t, y_old, *drift_args)`.
    drift_args :
        Pass `()` if no argument is needed.
    diffusion : Callable[[float, np.ndarray], float | np.ndarray]
        Diffusion coefficient in the SDE returning `float` or `np.ndarray`.
        Current time `t`, state variables `y_old` and other args are passed as `diffusion(t, y_old, *diff_args)`.
    diff_args : tuple
        Pass `()` if no argument is needed.
    t_calc : np.ndarray
        Times at which calculations are performed.
    y0 : np.ndarray
        Initial condition.
    t_eval : np.ndarray
        Times at which calculation results are saved and returned.
    rng : numpy.random.Generator, optional
        When `rng` is given, the passed `rng` is used.
        A new random generator with `seed` is generated otherwise.
    seed : int, optional
        `seed` must be passed unless `rng` is given.

    Returns
    -------
    res : Result
        `res` has following fields.
        t : numpy.ndarray
            Calculation points (times).
        y : numpy.ndarray
            Calculated time series (state variables).
    """
    # ---preparations---
    if t_eval is None:
        a_if_rec = np.ones(t_calc.shape, dtype="?")  # record flag
    else:
        assert np.isin(t_eval, t_calc).all(), "t_eval is not covered by t_calc!"
        a_if_rec = np.isin(t_calc, t_eval)  # record flag

    if rng is None:
        # create new rng with passed seed
        rng = np.random.default_rng(seed)

    n = y0.size  # dimension is inferred from initial condition
    a_y = np.empty((n, t_eval.size), dtype="f8")
    a_dt = t_calc[1:] - t_calc[:-1]

    # ---numerical integration---
    y_old: np.ndarray = y0.copy()
    rec_count: int = 0

    if a_if_rec[0]:
        a_y[:, 0] = y0.copy()
        rec_count += 1

    for if_rec, t, dt in zip(a_if_rec[1:], t_calc[:-1], a_dt):
        xi = rng.normal(loc=0, scale=np.sqrt(dt), size=n)
        y_new = (
            y_old
            + drift(t, y_old, *drift_args) * dt
            + diffusion(t, y_old, *diff_args) * xi
        )
        if if_rec:
            a_y[:, rec_count] = y_new.copy()
            rec_count += 1
        # update
        y_old = y_new.copy()

    # create bunch object
    res = Result()
    res.t = t_eval
    res.y = a_y

    return res
