# %%[markdown]
# # Figure 6
# - Panel a: K^{-1} dependence of the difference
# - Panel b: SD of displacements y_i

# %%
import numpy as np
from matplotlib import pyplot as plt

import gcbistablemet
from gcbistablemet.plot_utils import get_label_style, param_val_to_str, put_param_area
from gcbistablemet.utils import load_npz
from overall_utils import get_colors, get_markers, load_overall

# %%
# flags
if_show: bool = True  # If False, plt.show() won't be called.
if_save: bool = False  # If False, fig.savefig() won't be called.

# %%
# common style settings
plt.style.use("gcbistablemet.paperfigure")
t_col: tuple[str] = get_colors()
t_marker: tuple[str] = get_markers()

# %%
# load data for panel a
a_N = np.array([50, 100, 200], dtype="i8")
kid_min: int = 13

d_params_a, a_k, l_data = load_overall(a_N)

r = d_params_a["r"]
alpha = d_params_a["alpha"]
xi = d_params_a["xi"]
dt = d_params_a["dt"]
ss_t = d_params_a["ss_t"]

# %%
# load data for panel b
N_b: int = 200
d_params_b, a_k_b, a_meansdy, a_esc_count = load_npz(
    f"data/sdy/meansdy-N{N_b}.npz", allow_pickle=True
)
assert np.isclose(d_params_b["r"], r), "Different r value!"
assert np.isclose(d_params_b["alpha"], alpha), "Different alpha value!"
assert np.isclose(d_params_b["xi"], xi), "Different xi value!"
assert np.isclose(d_params_b["dt"], dt), "Different dt!"
assert np.isclose(d_params_b["ss_t"], ss_t), "Different ss_t!"

# %%
# hat{tau}_N
a_tauhatN = np.array([113.365, 174.776, 304.57], dtype="f8")
print(a_tauhatN)

# %%
# dict of style keyword args
ds_sim: dict = {"alpha": 0.9, "ls": "-", "lw": 1, "ms": 5}

# create figure
fig, (ax0, ax1) = plt.subplots(figsize=(7, 2.6), ncols=2)

# put panel labels
t_panlab = ("a", "b")
for ax, panlab in zip((ax0, ax1), t_panlab):
    ax.text(
        0.96,
        0.96,
        panlab,
        ha="right",
        va="top",
        transform=ax.transAxes,
        **get_label_style(),
    )

# --panel (a)--
# axes settings
ax0.set_xscale("log")
ax0.set_yscale("log")
ax0.set_ylim((3e-3, 500))
ax0.set_xlabel("coupling strength $K$")
ax0.set_ylabel(r"$\hat{\tau}_N - \overline{\tau}$")

# plot difference
for N_, data, tauhatN, c, m in zip(a_N, l_data, a_tauhatN, t_col, t_marker):
    a_met = data[2][kid_min:].mean(axis=2).mean(axis=1)
    ax0.plot(
        a_k[kid_min:], tauhatN - a_met, marker=m, c=c, **ds_sim, label=f"$N = {N_}$"
    )

# propto 1/K line
x = np.logspace(1.2, 3.2, 51, dtype="f8")
coeff: float = 80
ax0.plot(x, coeff / x, "k--")
ax0.text(700, coeff * 1e-3, "slope $= -1$", ha="right", va="top", size="large")

# put legend
ax0.legend(ncols=a_N.size, loc="lower center", bbox_to_anchor=(0.5, 1))

# --panel (b)--
ax1.set_title(f"$N = {N_b}$", loc="right", size="medium", pad=1)

ax1.errorbar(
    a_k_b,
    a_meansdy.mean(axis=1),
    yerr=a_meansdy.std(axis=1),
    c="k",
    ls="none",
    marker="o",
    ms=7,
    mfc="none",
    capsize=4,
    zorder=5,
    label="simulation",
)

# theory
x = np.logspace(-2, 4, 101, dtype="f8")
ax1.plot(
    x,
    alpha / np.sqrt(2 * x),
    c="tab:brown",
    ls=":",
    lw=2,
    zorder=2,
    label=r"$\alpha / \sqrt{2K}$",
)
ax1.plot(
    x,
    alpha / np.sqrt(2 * (x + r)),
    c="tab:pink",
    ls="-",
    lw=2,
    zorder=1,
    label=r"$\alpha / \sqrt{2(r + K)}$",
)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylim((3e-4, 0.5))
ax1.set_xlabel("coupling strength $K$")
ax1.set_ylabel("average SD of $y_i$")
ax1.legend(loc="lower left")

# put parameter values
d_paramstyle = put_param_area(fig)
l_nameval = [
    ("r", r),
    (r"\alpha", alpha),
    (r"\xi", xi),
    (r"\mathrm{dt}", dt),
    (r"\Delta t", ss_t),
]

l_nameval_a = [(r"n_{\, \mathrm{trial}}", d_params_a["n_trial"])]
for N_, tauhatN_ in zip(a_N, a_tauhatN):
    l_nameval_a.append((rf"\hat{{\tau}}_{{{N_}}}", tauhatN_))

concat = lambda l_tup: ", ".join([param_val_to_str(tup[0], tup[1]) for tup in l_tup])
s_common: str = concat(l_nameval)
s_a: str = concat(l_nameval_a)
s_b: str = param_val_to_str(r"n_{\, \mathrm{trial}}", d_params_b["n_trial"])
s_b += rf", $t \in [0, {d_params_b['t_max']}]$"

s: str = s_common + "\npanel (a): " + s_a + "; panel (b): " + s_b

fig.text(s=s, **d_paramstyle)

# show & save
if if_show == True:
    plt.show()
if if_save == True:
    fig.savefig("largek.pdf")
# %%
