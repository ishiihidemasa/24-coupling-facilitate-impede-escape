# calculate data for figure 6(b)
import numpy as np
from tqdm import trange

from gcbistablemet.gc_bistable import sim_gc_bistable

N = 200
r = 0.05
alpha = 0.1
a_k = np.logspace(-2, 4, 13, dtype="f8")
t_max = 10
dt: float = 2e-5
ss_t: float = 1e-3
n_trial = 100
xi = 0.5

a_meansdy = np.empty((a_k.size, n_trial), dtype="f8")
esc_count = np.empty((a_k.size, n_trial), dtype="i8")

for j, k in enumerate(a_k):
    for s in trange(n_trial):
        rng = np.random.default_rng(seed=s)
        res = sim_gc_bistable(
            r, k, alpha, 0, t_max, dt, np.zeros(N, dtype="f8"), ss_t, rng
        )
        esc_count[j, s] = (res.y[:, -1] >= xi).sum()

        mf = res.y.mean(axis=0, keepdims=True)  # mean field
        yi = res.y - mf  # displacement
        a_meansdy[j, s] = np.std(yi, axis=0).mean()

d_params = {
    "N": N,
    "r": r,
    "alpha": alpha,
    "xi": xi,
    "t_max": t_max,
    "dt": dt,
    "ss_t": ss_t,
    "n_trial": n_trial,
}

np.savez_compressed(
    f"data/sdy/meansdy-N{N}.npz",
    a_d_params=np.array([d_params], dtype="O"),
    a_k=a_k,
    a_meansdy=a_meansdy,
    a_esccount=esc_count,
)

# show the number of trials where some nodes escaped
print(np.sum(esc_count > 0, axis=1))
