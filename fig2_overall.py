# %%[markdown]
# # Figure 2

# %%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
from scipy.stats import norm

import gcbistablemet
from gcbistablemet.gc_bistable import mfpt, mfpt_kramers
from gcbistablemet.plot_utils import get_label_style, param_val_to_str, put_param_area
from gcbistablemet.utils import load_npz
from overall_utils import get_colors, get_markers, load_overall

# %%
a_N = np.array([2, 50, 100, 150, 200], dtype="i8")
l_id_overall = [1, 2, 4]

# flags
if_show: bool = True  # If False, plt.show() won't be called.
if_save: bool = False  # If False, fig.savefig() won't be called.

plt.style.use("gcbistablemet.paperfigure")

# %%
# create figure with vertically stacked three subfigures
fig = plt.figure(figsize=(7, 6.5))
subfig_all = fig.subfigures(nrows=4, height_ratios=(16, 15, 2, 14))
subfigs = subfig_all[[0, 1, 3]]

# %%[markdown]
# ## Panels a & b
# Relation between coupling strength and mean escape time

# %%
# prepare data
a_N_overall = a_N[l_id_overall]

d_params, a_k, l_data = load_overall(a_N_overall)

r = d_params["r"]
alpha = d_params["alpha"]
xi = d_params["xi"]

# %%
# common style settings
t_col: tuple[str] = get_colors()
t_marker: tuple[str] = get_markers()

# %%
# flags
if_sd: bool = False  # If True, standard deviation of AET is shown as well.

# dict of style keyword args
ds_sim: dict = {"alpha": 0.7, "mfc": "none", "ms": 7, "mew": 2, "ls": ":", "lw": 3}
ds_fill: dict = {"alpha": 0.2, "zorder": 1, "edgecolor": "none"}
ds_mfpt: dict = {"ls": "--", "zorder": 2}

k_sep: int = 13  # border between linear & log scales

# create subplots
ax0, ax1 = subfigs[0].subplots(ncols=2, width_ratios=(k_sep, a_k.size - k_sep))

# put panel labels
t_panlab = ("a", "b")
for ax, panlab in zip((ax0, ax1), t_panlab):
    ax.text(
        0.98,
        0.02,
        panlab,
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        **get_label_style(),
    )

# prepare axes
ax0.set_ylabel(r"mean escape time $\overline{\tau}$")
ax1.yaxis.tick_right()
ax1.set_xscale("log")
for ax in (ax0, ax1):
    ax.set_xlabel("coupling strength $K$")
if not if_sd:
    ax0.set_ylim((12.5, 21.5))
    ax1.set_ylim((0, 325))

# plot simulation results
l_lines = []

for N, data, c, m in zip(a_N_overall, l_data, t_col, t_marker):
    a_aet_ = data[2].mean(axis=2)
    a_met_ = a_aet_.mean(axis=1)

    # mean escape time
    (l_,) = ax0.plot(
        a_k[:k_sep], a_met_[:k_sep], c=c, marker=m, label=f"$N = {N}$", **ds_sim
    )
    ax1.plot(a_k[k_sep:], a_met_[k_sep:], c=c, marker=m, **ds_sim)

    l_lines.append(l_)  # for legend

    if if_sd == True:
        # standard deviation
        a_sd_aet = a_aet_.std(axis=1)
        # a_sd_aet = a_sd_aet / d_params["n_trial"]**0.5  # Standard Error of the Mean

        ax0.fill_between(
            a_k[:k_sep],
            y1=(a_met_ - a_sd_aet)[:k_sep],
            y2=(a_met_ + a_sd_aet)[:k_sep],
            color=c,
            **ds_fill,
        )
        ax1.fill_between(
            a_k[k_sep:],
            y1=(a_met_ - a_sd_aet)[k_sep:],
            y2=(a_met_ + a_sd_aet)[k_sep:],
            color=c,
            **ds_fill,
        )

    # theoretical value
    t_infty = mfpt(r, alpha / N**0.5, xi)
    ax1.axhline(t_infty, c=c, **ds_mfpt)
    ax1.text(
        0.3,
        t_infty - 6,
        rf"$T_{{\infty}}({N})$",
        ha="left",
        va="top",
    )

# T_0
met0 = mfpt(r, alpha, xi)
ax0.axhline(met0, c="k", **ds_mfpt)
ax0.text(
    0.28,
    met0 - 0.3,
    rf"$T_0$",
    ha="right",
    va="top",
)

# put legend
subfigs[0].legend(
    handles=l_lines, ncols=a_N_overall.size, loc="lower center", bbox_to_anchor=(0.5, 1)
)

# %%[markdown]
# ## Panel c
# N dependence under (almost) infinitely strong coupling

# %%
# prepare data
a_met = np.empty(a_N.size, dtype="f8")
a_sd_aet = np.empty(a_N.size, dtype="f8")

for nid, data in zip(l_id_overall, l_data):
    a_aet = data[2][-1].mean(axis=1)
    a_met[nid] = a_aet.mean(axis=0)
    a_sd_aet[nid] = a_aet.std(axis=0)

for nid in (0, 3):
    N_ = a_N[nid]
    fname_ = f"n{N_}-r005-alpha010-k10000_0000-nt2000.npz"
    data = load_npz("data/overall/" + fname_, allow_pickle=True)

    d_ = dict(data[0])
    assert N_ == d_.pop("N")

    try:
        assert d_ == d_params
    except AssertionError as ae:
        print("others:\n", d_params)
        print(f"current (N == {N_}):\n", d_)
        raise ae

    a_aet = data[2].squeeze().mean(axis=1)
    a_met[nid] = a_aet.mean(axis=0)
    a_sd_aet[nid] = a_aet.std(axis=0)

# %%
# prepare theoretical values
x = np.linspace(a_N[0], a_N[-1], 101)
y = np.array([mfpt(r, alpha / n.item() ** 0.5, xi) for n in x])
y_kra = np.array([mfpt_kramers(r, alpha / n.item() ** 0.5) for n in x])

# %%
# create subplots
axes = subfigs[1].subplots(ncols=3, width_ratios=(1, 2, 1))

# axes settings
axes[0].set_visible(False)
axes[2].set_visible(False)
ax = axes[1]

# put panel label
ax.text(
    0.02, 0.96, "c", ha="left", va="top", transform=ax.transAxes, **get_label_style()
)

# axes settings
ax.set_xlabel("the number of nodes $N$")
ax.set_ylabel("average escape time\n" + r"$\langle \tau_i \rangle$")
ax.set_yscale("log")
ax.set_xticks(a_N)
ax.set_ylim((2, 800))

# plot numerical results
ax.errorbar(
    a_N,
    a_met,
    yerr=a_sd_aet,
    c="k",
    marker="o",
    mfc="none",
    ms=6,
    mew=1.5,
    ls="none",
    capsize=3,
    label=r"mean $\pm$ SD",
    zorder=5,
)

# plot theoretical values
ax.plot(x, y, "tab:blue", alpha=0.8, ls="--", label=r"$T_{\infty}(N)$")
ax.plot(x, y_kra, "tab:red", alpha=0.8, ls=":", label=r"$\tilde{T}_{\infty}(N)$")

# put legend
ax.legend(ncols=2)

# %%[markdown]
# ## Panels d, e, & f
# Histogram for average escape times

# %%
k_ids: list[int] = [0, 15, 31]
n_id = 2
d_params, a_k, a_fpt, _ = l_data[n_id]
N = a_fpt.shape[-1]

t_panlab = ("d", "e", "f")

# create subplots
axes = subfigs[2].subplots(ncols=len(k_ids))

# axes settings
axes[0].set_title(f"$N = {N}$", loc="left", size="medium", pad=1)
axes[0].set_ylabel("density")

# plot histogram
l_x = []

for i, (ax, panlab) in enumerate(zip(axes.flatten(), t_panlab)):
    k_id = k_ids[i]
    k = a_k[k_id]
    arr = a_fpt[k_id]

    # panel label
    ax.text(
        0.95,
        0.96,
        panlab,
        ha="right",
        va="top",
        transform=ax.transAxes,
        **get_label_style(),
    )
    # show K value
    assert k.is_integer(), "K is not an integer!"
    s: str = f"$K = {k.astype('i8')}$"
    ax.text(0.95, 0.75, s, ha="right", va="top", transform=ax.transAxes)

    # x label
    ax.set_xlabel(r"average escape time $\langle \tau_i \rangle$")

    # histogram
    n, b, p = ax.hist(
        arr.mean(axis=1),
        bins="auto",  # let numpy determine the number of bins
        density=True,
        color="lightgrey",
        ec="k",
        histtype="stepfilled",
    )
    ax.set_ylim((0, 1.05 * n.max()))

    # calculate mean and SD
    x = np.linspace(b[0], b[-1], 201)
    l_x.append(x)

# PDF curves
## normal dist.
x = l_x[0]
mu = mfpt(d_params["r"], d_params["alpha"], d_params["xi"])
sd = mu / N**0.5
(ln,) = axes[0].plot(
    x,
    norm.pdf(x, loc=mu, scale=sd),
    c=t_col[0],
    alpha=0.8,
    ls="--",
    lw=3,
    zorder=3,
    label=rf"$\mathcal{{N}} \, ({mu:.1f}, {sd * sd:.2f})$",
)

## exponential dist.
x = l_x[-1]
mu = mfpt(d_params["r"], d_params["alpha"] / N**0.5, d_params["xi"])

lfsn = LogFormatterSciNotation()
s_mu: str = lfsn(round(1 / mu, 5))[1:-1]

(le,) = axes[-1].plot(
    x,
    np.exp(-x / mu) / mu,
    c=t_col[2],
    alpha=0.8,
    ls="-",
    lw=3,
    zorder=3,
    label=rf"$\mathrm{{Exp}} \, ({s_mu})$",
)

# put legend
subfigs[2].legend(
    handles=[ln, le],
    ncols=2,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.96),
)

# show parameter values
d_paramstyle = put_param_area(fig)
l_nameval = [
    ("r", r),
    (r"\alpha", alpha),
    (r"\xi", xi),
    (r"\mathrm{dt}", d_params["dt"]),
    (r"\Delta t", d_params["ss_t"]),
    (r"n_{\, \mathrm{trial}}", d_params["n_trial"]),
]

fig.text(
    s=", ".join([param_val_to_str(tup[0], tup[1]) for tup in l_nameval]), **d_paramstyle
)

# show and save
if if_show == True:
    plt.figure(fig)
    plt.show()
if if_save == True:
    fig.savefig("overall-results.pdf")

# %%
