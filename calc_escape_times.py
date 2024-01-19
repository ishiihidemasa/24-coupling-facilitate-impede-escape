# this script was run on HPC with torque (job scheduler)
# %%
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "10"  # required to run this script on HPC
import numpy as np

from gcbistablemet.gc_bistable import calc_escape_times
from gcbistablemet.utils import get_fname

# possible parameter values
a_N = np.array([2, 150, 50, 100, 200], dtype="i8")
a_alpha = np.array([0.02, 0.1], dtype="f8")
a_k_all = np.concatenate(
    (
        25e-3 * np.arange(13, dtype="f8"),  # [0, 0.3]
        np.logspace(-1 / 2, 4, 19),  # (0.3, 1e4]
        1e-2 * np.arange(8, dtype="f8"),  # [0, 0.07]  for alpha = 0.2
    )
)

r = 0.05
xi = 0.5

# %%
# calculation settings
n_trial = 2000
per_epoch = 50
a_so = np.arange(-(n_trial // -per_epoch), dtype="i8") * per_epoch

dt = 2e-5
ss_t = 1e-3
t_epoch = 2
max_epoch = 25000

# set parameter values from command line inputs
args = sys.argv  # (script path, N, alpha, K, seed_offset, savedir)
if_digit = np.array([arg.isdigit() for arg in args[1:-1]], dtype="?").all()
assert if_digit, "First four arguments must be digits!"

# N
i = int(args[1])
N = a_N[i]
# alpha
j = int(args[2])
alpha = a_alpha[j]
# K
l = int(args[3])
a_k = a_k_all[l : l + 1]
# seed_offset
m = int(args[4])
so = a_so[m]
# savedir
savedir = args[5]

# calculate and save results
fname: str = get_fname(N, r, alpha, a_k, so)
print(fname)

data_ = calc_escape_times(
    N,
    r,
    alpha,
    a_k,
    dt,
    ss_t,
    xi,
    n_trial,
    t_epoch,
    max_epoch,
    so,
    savedir=savedir,
    fname=fname,
)
