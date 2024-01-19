# calculate data for figures 3 & 4
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

from gcbistablemet.weak_coupling_theory import dblfpe, fpe

# model parameters
r: float = 0.1
alpha: float = 0.1
a_k = np.array([0, 0.01, 0.2], dtype="f8")

# simulation settings
a, b = (-0.5, 1.4)  # when (r, alpha) = (0.1, 0.1)
ss_x = 1e-3  # step size (space)
t_max = 45
ss_t = 1e-1  # step size (time)

a_time = ss_t * np.arange(round(t_max / ss_t) + 1, dtype="f8")

num_point = np.round((b - a) / ss_x).astype("i8") + 1
x_ext = a - ss_x + ss_x * np.arange(num_point + 2, dtype="f8")

# calculate
for k in tqdm(a_k):
    k: float = k.item()
    # print(f"r = {r}, alpha = {alpha}, K = {k}")

    d_params = {
        "r": r,
        "alpha": alpha,
        "k": k,
        "a": a,
        "b": b,
        "ss_x": ss_x,
        "t_max": t_max,
        "ss_t": ss_t,
    }

    init = np.zeros(num_point, dtype="f8")
    init[np.round(-a // ss_x).astype("i8")] = 1 / ss_x

    if k == 0:
        # simulate FPE for K = 0
        res = solve_ivp(
            fpe,
            (0, t_max),
            init,
            args=(r, alpha, x_ext, ss_x),
            t_eval=a_time,
            atol=1e-8,
            rtol=1e-8,
        )

    else:
        # for K > 0
        res = solve_ivp(
            dblfpe,
            (0, t_max),
            np.hstack((init, init)),
            args=(r, k, alpha, x_ext, ss_x),
            t_eval=a_time,
            atol=1e-8,
            rtol=1e-8,
        )

    fname = f"r{round(r*1e2):03}-alpha{round(alpha*1e2):03}-k{round(k*1e3):04}.npz"
    np.savez_compressed(
        "data/weaktheory/" + fname,
        x_ext=x_ext,
        res_t=res.t,
        res_y=res.y,
        a_d_params=np.array([d_params], dtype="O"),
    )
