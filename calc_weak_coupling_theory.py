# this script was run on HPC with torque (job scheduler)
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import trange

from gcbistablemet.weak_coupling_theory import dblfpe, fpe

os.environ["OPENBLAS_NUM_THREADS"] = "5"  # required to run this script on HPC

# model parameters from command line inputs
a_r = np.array([0.05, 0.1], dtype="f8")
a_k = np.hstack(
    (
        np.array([0, 0.01, 0.2], dtype="f8"),  # to illustrate mean field evolution
        1e-2 * np.arange(7, dtype="f8"),  # (r, alpha) = (0.05, 0.02) case
        5e-2 * np.arange(7, dtype="f8"),  # (r, alpha) = (0.05, 0.1) case
    )
)
a_alpha = np.array([0.02, 0.1], dtype="f8")

args = sys.argv
# r
r = a_r[int(args[1])].item()
# alpha
alpha = a_alpha[int(args[2])].item()
# k
k = a_k[int(args[3])].item()

print(f"r = {r}, alpha = {alpha}, K = {k}")

# simulation settings
ss_x = 0.001  # step size (space)
t_epoch = 2
max_epoch = 1000
mf_end = 0.99
ss_t = 1e-1  # step size (time)

if r == 0.1 and alpha == 0.1:
    a, b = (-0.5, 1.4)
elif alpha == 0.1:
    # (r, alpha) = (0.05, 0.1) -> MET = 21.9
    a, b = (-0.5, 1.3)
elif alpha == 0.02:
    # (r, alpha) = (0.05, 0.02) -> MET = 79.5
    a, b = (-0.2, 1.2)

num_point = np.round((b - a) / ss_x).astype("i8") + 1
x_ext = a - ss_x + ss_x * np.arange(num_point + 2, dtype="f8")

d_params = {
    "r": r,
    "alpha": alpha,
    "k": k,
    "a": a,
    "b": b,
    "ss_x": ss_x,
    "t_epoch": t_epoch,
    "max_epoch": max_epoch,
    "ss_t": ss_t,
    "mf_end": mf_end,
}

if k == 0:
    # simulate FPE for K = 0
    init = np.zeros(num_point, dtype="f8")
    init[np.round(-a // ss_x).astype("i8")] = 1 / ss_x

    p_last = init.copy()
    t_last = 0
    for i in trange(max_epoch):
        a_time = i * t_epoch + ss_t * np.arange(round(t_epoch / ss_t), dtype="f8")

        res = solve_ivp(
            fpe,
            (t_last, (i + 1) * t_epoch),
            p_last,
            args=(r, alpha, x_ext, ss_x),
            t_eval=a_time,
            atol=1e-8,
            rtol=1e-8,
        )
        # update
        t_last = res.t[-1]
        p_last = res.y[:, -1]
        if i == 0:
            res_t = res.t.copy()
            res_y = res.y.copy()
        else:
            res_t = np.hstack((res_t, res.t.copy()))
            res_y = np.hstack((res_y, res.y.copy()))

        # terminate if X >= terminate is achieved
        mf_last = ss_x * np.sum(x_ext[1:-1] * res_y[:, -1])
        if mf_last >= mf_end:
            break

else:
    # for K > 0
    init = np.zeros(num_point, dtype="f8")
    init[np.round(-a // ss_x).astype("i8")] = 1 / ss_x

    p_last = np.hstack((init, init))
    t_last = 0
    for i in trange(max_epoch):
        a_time = i * t_epoch + ss_t * np.arange(round(t_epoch / ss_t), dtype="f8")
        res = solve_ivp(
            dblfpe,
            (t_last, (i + 1) * t_epoch),
            p_last,
            args=(r, k, alpha, x_ext, ss_x),
            t_eval=a_time,
            atol=1e-8,
            rtol=1e-8,
        )
        # update
        t_last = res.t[-1]
        p_last = res.y[:, -1]
        if i == 0:
            res_t = res.t.copy()
            res_y = res.y.copy()
        else:
            res_t = np.hstack((res_t, res.t.copy()))
            res_y = np.hstack((res_y, res.y.copy()))

        # terminate if X >= terminate is achieved
        mf_last = ss_x * np.sum(x_ext[1:-1] * res_y[num_point:, -1])
        if mf_last >= mf_end:
            break

fname = f"r{round(r*1e2):03}-alpha{round(alpha*1e2):03}-k{round(k*1e3):04}.npz"
savedir = "weaktheory/"
os.makedirs("output/" + savedir, exist_ok=True)  # make sure the directory exists

np.savez_compressed(
    "output/" + savedir + fname,
    x_ext=x_ext,
    res_t=res_t,
    res_y=res_y,
    a_d_params=np.array([d_params], dtype="O"),
)
