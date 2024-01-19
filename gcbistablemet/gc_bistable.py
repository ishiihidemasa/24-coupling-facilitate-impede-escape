# globally-coupled bistable models
import math
import os

import numpy as np
from scipy.integrate import dblquad
from tqdm import tqdm, trange

from gcbistablemet.integrate import euler_maruyama


def schloegl(t: float, x: np.ndarray, r: float) -> np.ndarray:
    """
    f(x) = -x (x - r) (x - 1)
    """
    return -x * (x - r) * (x - 1)


def gc_drift(t: float, x: np.ndarray, r: float, k: float) -> np.ndarray:
    """Drift term of SDE under mean-field approximation."""
    mf: float = x.mean()
    return schloegl(t, x, r) + k * (mf - x)


def mfpt(
    r: float, alpha: float | list[float] | np.ndarray, xi: float, init: float = 0
) -> float | np.ndarray:
    """
    Parameter
    ---------
    r : float
    alpha : float | list[float] | np.ndarray
        List of float values is not expected.
    xi : float
    init: float

    Returns
    -------
    float | np.ndarray
    """
    # make sure alpha is a float or a list of float
    if type(alpha) == float:
        num_alpha = 1
    else:
        # alpha is either a list of float or ndarray
        try:
            num_alpha = alpha.size
            alpha = alpha.tolist()
        except AttributeError:
            num_alpha = len(alpha)

    # integrand
    ## original: np.exp(2 * potf(x) / alpha**2) * np.exp(-2 * potf(y) / alpha**2
    ## where: potf = lambda v: v**4 / 4 - (1 + r) / 3 * v**3 + r * v * v / 2
    # fmt: off
    integrand = lambda y, x, a: math.e ** (2 / a**2 * (
            (x**4 - y**4) / 4
            - (1 + r) / 3 * (x**3 - y**3)
            + r / 2 * (x**2 - y**2)
        )
    )
    # fmt: on

    if num_alpha == 1:
        # return a float
        fun = lambda y, x: integrand(y, x, alpha)
        num_int, err = dblquad(fun, init, xi, lambda x: -np.inf, lambda x: x)
        t = 2 / alpha**2 * num_int

    else:
        # return an array
        t = np.empty(shape=num_alpha, dtype="f8")
        for i, alpha_ in enumerate(alpha):
            fun = lambda y, x: integrand(y, x, alpha_)
            num_int, err = dblquad(fun, init, xi, lambda x: -np.inf, lambda x: x)
            t[i] = 2 / alpha_**2 * num_int

    return t


def mfpt_kramers(r: float, alpha: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates mean first passage time for 1D system using Kramers formula.
    """
    tk = 2 * np.pi / (r * np.sqrt(1 - r)) * np.exp((2 - r) * r**3 / (6 * alpha**2))
    return tk


def sim_gc_bistable(
    r: float,
    k: float,
    alpha: float,
    t0: float,
    t_sim: float,
    dt: float,
    y0: np.ndarray,
    ss_t: float,
    rng: np.random.Generator,
):
    t_calc = t0 + dt * np.arange(round(t_sim / dt) + 1, dtype="f8")
    rem = ss_t - dt * int(ss_t / dt)
    assert np.isclose(rem, 0), "ss_t % dt must be 0!"
    interval = round(ss_t / dt)
    t_eval = t_calc[::interval]

    if k == 0:
        res = euler_maruyama(
            schloegl,
            (r,),
            lambda t, x: alpha,
            (),
            t_calc,
            y0,
            t_eval,
            rng=rng,
        )
    else:
        res = euler_maruyama(
            gc_drift,
            (r, k),
            lambda t, x: alpha,
            (),
            t_calc,
            y0,
            t_eval,
            rng=rng,
        )
    return res


def calc_escape_times(
    N: int,
    r: float,
    alpha: float,
    a_k: np.ndarray,
    dt: float,
    ss_t: int,
    xi: float,
    n_trial: int,
    t_epoch: float,
    max_epoch: int,
    seed_offset: int,
    save: bool = True,
    savedir: str = None,
    fname: str = None,
    terminate_prop: float = 1,
    y0: None | np.ndarray = None,
):
    d_params = {
        "N": N,
        "r": r,
        "alpha": alpha,
        "dt": dt,
        "ss_t": ss_t,
        "xi": xi,
        "n_trial": n_trial,
        "t_epoch": t_epoch,
        "max_epoch": max_epoch,
        "seed_offset": seed_offset,
        "terminate_prop": terminate_prop,
    }

    a_fpt: np.ndarray = -np.ones(shape=(a_k.size, n_trial, N + 1), dtype="f8")

    if y0 is None:
        y0 = np.zeros(N, dtype="f8")

    for i, k in tqdm(enumerate(a_k)):
        for j in range(n_trial):
            elapsed: float = 0  # time within one trial

            rng = np.random.default_rng(seed=seed_offset + j)
            init_epoch: np.ndarray = y0.copy()

            for epoch in trange(max_epoch):
                res = sim_gc_bistable(
                    r, k, alpha, elapsed, t_epoch, dt, init_epoch, ss_t, rng
                )
                # --record times when ts_prop exceeded prop--
                # extract values to update
                data = np.concatenate(
                    (res.y, res.y.mean(axis=0, keepdims=True)), axis=0
                )
                if_passed = data >= xi
                # values have not been updated & x_i passed threshold
                mask = (a_fpt[i, j] < 0) * if_passed.any(axis=1)
                if mask.any():
                    # extract first time when True appears in 1d boolean array
                    first_true = lambda arr_bool, t: t[arr_bool][0]
                    a_fpt[i, j, mask] = np.apply_along_axis(
                        first_true, 1, if_passed[mask], res.t
                    )

                # if most nodes reached active state, stop loop.
                num_passed = (a_fpt[i, j, :-1] >= 0).sum()
                if num_passed / N >= terminate_prop:
                    break

                # update initial condition
                init_epoch = res.y[:, -1]
                elapsed = t_epoch * (epoch + 1)

            # when one calculation is over
            else:
                print(f"terminate_prop was not achieved: K = {k.round(10)}")

    a_mf_fpt = a_fpt[:, :, -1]
    a_fpt = a_fpt[:, :, :-1]

    if save:
        if len(savedir) == 0:
            outdir: str = "output"
        else:
            outdir: str = "output/" + savedir

        os.makedirs(outdir, exist_ok=True)  # make sure the directory exists
        np.savez_compressed(
            outdir + "/" + fname,
            a_d_params=np.array([d_params], dtype="O"),
            a_k=a_k,
            a_fpt=a_fpt,
            a_mf_fpt=a_mf_fpt,
        )

    return d_params, a_k, a_fpt, a_mf_fpt
