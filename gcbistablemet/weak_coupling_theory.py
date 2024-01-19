import sys

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm, trange


def _get_pext(x_ext, p):
    p_ext = np.empty(x_ext.size, dtype="f8")  # extended
    p_ext[[0, -1]] = p[[2, -3]]  # Neumann BC
    p_ext[[1, -2]] = 0  # Dirichlet BC
    p_ext[2:-2] = p[1:-1]
    return p_ext


def fpe(t, p, r, alpha, x_ext, h):
    f = lambda x: -x * (x - r) * (x - 1)
    dfdx = lambda x: -3 * x * x + 2 * (1 + r) * x - r
    p_ext = _get_pext(x_ext, p)

    # fmt: off
    pdot = (
        f(x_ext[1: -1]) * (p_ext[:-2] - p_ext[2:]) / (2 * h) - p_ext[1: -1] * dfdx(x_ext[1:-1])
        + alpha * alpha / 2 * (
        p_ext[2:] + p_ext[:-2] - 2 * p_ext[1:-1]
    ) / (h * h)
    )
    # fmt: on
    return pdot


def dblfpe(t, p, r, k, alpha, x_ext, h):
    """
    p[:num_point] is PDF for K = 0, and p[num_point:] is PDF for K > 0.
    """
    num_point = p.size // 2  # the number of calculation points
    f = lambda x: -x * (x - r) * (x - 1)
    dfdx = lambda x: -3 * x * x + 2 * (1 + r) * x - r

    mf0 = h * np.sum(x_ext[1:-1] * p[:num_point])

    p0_ext = _get_pext(x_ext, p[:num_point])
    pk_ext = _get_pext(x_ext, p[num_point:])

    # fmt: off
    p0dot = (
        f(x_ext[1: -1]) * (p0_ext[:-2] - p0_ext[2:]) / (2 * h) - p0_ext[1: -1] * dfdx(x_ext[1:-1])
        + alpha * alpha / 2 * (
        p0_ext[2:] + p0_ext[:-2] - 2 * p0_ext[1:-1]
    ) / (h * h)
    )
    pkdot = (
        (f(x_ext[1: -1]) + k * (mf0 - x_ext[1:-1])) * (pk_ext[:-2] - pk_ext[2:]) 
        / (2 * h) 
        - pk_ext[1: -1] * (dfdx(x_ext[1:-1]) - k)
        + alpha * alpha / 2 * (
        pk_ext[2:] + pk_ext[:-2] - 2 * pk_ext[1:-1]
    ) / (h * h)
    )
    # fmt: on
    return np.hstack((p0dot, pkdot))


def prob_current(res_y, r, alpha, k, xi, x_ext, a, ss_x):
    """Calculate probability current at x = xi."""
    num_point = x_ext.size - 2
    offset = 0 if k == 0 else num_point
    xext_id = np.round((xi - a) / ss_x).astype("i8") + 1

    y_id = offset + xext_id - 2
    x = x_ext[xext_id - 1 : xext_id + 2]  # midpoint is xi
    fx = -x * (x - r) * (x - 1)
    dpdx = (res_y[y_id : y_id + 3] - res_y[y_id - 2 : y_id + 1]) / (2 * ss_x)
    if k == 0:
        current = fx[:, None] * res_y[y_id - 1 : y_id + 2] - alpha * alpha / 2 * dpdx
    else:
        mf0 = ss_x * np.sum(res_y[:num_point] * x_ext[1:-1, None], axis=0)
        current = (fx[:, None] + k * (mf0[None] - x[:, None])) * res_y[
            y_id - 1 : y_id + 2
        ] - alpha * alpha / 2 * dpdx  # (position, time)
    return current
