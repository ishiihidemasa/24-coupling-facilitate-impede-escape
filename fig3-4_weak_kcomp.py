# %%
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import gcbistablemet
from gcbistablemet.gc_bistable import sim_gc_bistable
from gcbistablemet.plot_utils import get_label_style, param_val_to_str, put_param_area
from gcbistablemet.utils import load_npz

# %%
# flags
if_show: bool = True  # If False, plt.show() won't be called.
if_save: bool = False  # If False, fig.savefig() won't be called.

# %%
# common style settings
plt.style.use("gcbistablemet.paperfigure")

# %%[markdown]
# # Figure 3
# Compare mean field trajectories
# - Panels b, d: results of direct numerical simulations
# - Panels c, e: results of our theory

# %%
r, alpha = (0.1, 0.1)
a_k = np.array([0, 0.01, 0.2], dtype="f8")

tlast_id: int = 300

# %%
# data for panels (c, e) [theory]
l_data = []

for k in tqdm(a_k):
    fname = (
        f"r{round(r*1e2):03}-alpha{round(alpha*1e2):03}-k{round(k.item()*1e3):04}.npz"
    )
    data = load_npz("data/weaktheory/" + fname, allow_pickle=True)
    l_data.append(data)
    d_ = dict(data[0])
    assert np.isclose(k, d_.pop("k"))
    try:
        assert d_ == d_params
    except NameError:
        d_params = d_

    try:
        assert (x_ext == data[1]).all()
    except NameError:
        x_ext = data[1]

a = d_params["a"]
b = d_params["b"]
ss_x = d_params["ss_x"]
num_point = x_ext.size - 2

ss_t = d_params["ss_t"]
t_max = d_params["t_max"]

# %%
# data for panels (b, d) [direct simulation]
N = 5000
dt = 1e-3
seed = 0

l_res = []
for k in tqdm(a_k):
    rng = np.random.default_rng(seed=seed)
    l_res.append(
        sim_gc_bistable(
            r, k.item(), alpha, 0, t_max, dt, np.zeros(N, dtype="f8"), ss_t, rng
        )
    )

# %%
# plot
t_c = ("k", "tab:cyan", "tab:orange")
t_klab = ("$K = 0$", "$K = r^2$", "$K = 2r$")
t_panlab = ("b", "c", "d", "e")
d_ls0 = {"lw": 1, "ls": "--", "zorder": 5}
d_lsk = {"lw": 2, "alpha": 0.8}

fig, axes = plt.subplots(
    ncols=2,
    nrows=2,
    figsize=(7, 2.6),
    height_ratios=(3, 2),
)

# show panel labels
for ax, panlab in zip(axes.flatten(), t_panlab):
    ax.text(
        0.02,
        0.96,
        panlab,
        ha="left",
        va="top",
        transform=ax.transAxes,
        **get_label_style(),
    )

d_desc_style = {"x": 0.1, "y": 0.92, "ha": "left", "va": "top"}
for ax in axes[:, 0]:
    ax.text(s="simulation", transform=ax.transAxes, **d_desc_style)
for ax in axes[:, 1]:
    ax.text(s="theory", transform=ax.transAxes, **d_desc_style)

# axes settings
ax0, ax1, ax2, ax3 = axes.flatten()
# tick settings
ax1.sharey(ax0)
ax3.sharey(ax2)
ax2.sharex(ax0)
ax2.set_xlim((0, t_max))
ax3.sharex(ax1)
ax3.set_xlim((0, t_max))
for ax in axes[:, 1]:
    ax.tick_params(left=False, labelleft=False)
for ax in axes[0, :]:
    ax.tick_params(bottom=False, labelbottom=False)

# axes labels
ax0.set_ylabel("mean field $X_K(t)$")
ax2.set_ylabel("$X_K - X_0$")
for ax in axes[1, :]:
    ax.set_xlabel("time $t$")
    ax.set_xlim((0, ss_t * tlast_id))

# plot data
l_lines = []

## SDE
# K = 0
mf0_sde = l_res[0].y.mean(axis=0)
(l_,) = ax0.plot(l_res[0].t, mf0_sde, c=t_c[0], label=t_klab[0], **d_ls0)
ax2.axhline(0, c=t_c[0], **d_ls0)
l_lines.append(l_)

# K > 0
for k, res, c, lab in zip(a_k[1:], l_res[1:], t_c[1:], t_klab[1:]):
    mf = res.y.mean(axis=0)
    (l_,) = ax0.plot(res.t, mf, c=c, label=lab, **d_lsk)
    ax2.plot(res.t, (mf - mf0_sde), c=c, **d_lsk)
    l_lines.append(l_)

## theory
for k, data, c, lab in zip(a_k, l_data, t_c, t_klab):
    if k == 0:
        mf0_fpe = (
            ss_x
            * np.sum(x_ext[None, 1:-1] * data[3][:num_point].T, axis=1)[: tlast_id + 1]
        )
        ax1.plot(data[2][: tlast_id + 1], mf0_fpe, c=c, label=lab, **d_ls0)
        ax3.axhline(0, c=t_c[0], **d_ls0)
    else:
        mf = (
            ss_x
            * np.sum(x_ext[None, 1:-1] * data[3][num_point:].T, axis=1)[: tlast_id + 1]
        )
        ax1.plot(data[2][: tlast_id + 1], mf, c=c, label=lab, **d_lsk)
        ax3.plot(data[2][: tlast_id + 1], (mf - mf0_fpe), c=c, **d_lsk)

# put legend
fig.legend(handles=l_lines, ncols=a_k.size, loc="lower center", bbox_to_anchor=(0.5, 1))

# show parameter values
d_paramstyle = put_param_area(fig)
l_nameval = [
    ("r", r),
    (r"\alpha", alpha),
    ("a", d_params["a"]),
    ("b", d_params["b"]),
    (r"\Delta x", d_params["ss_x"]),
]

fig.text(
    s=", ".join([param_val_to_str(tup[0], tup[1]) for tup in l_nameval]), **d_paramstyle
)

if if_show == True:
    plt.show()
if if_save == True:
    fig.savefig("weak-mfcomp.pdf")

# %%[markdown]
# ## Figure 4
# Compare snapshots of PDFs

# %%
a_tid = np.array([0, 100, 200, 450], dtype="i8")
t_panlab = ("a", "b", "c")
l_col = plt.get_cmap("plasma_r")(np.linspace(0.1, 0.75, a_tid.size))

# create figure
fig, axes = plt.subplots(figsize=(7, 2), ncols=3, sharex=True, sharey=True)

# plot data
l_lines = []

for k, data, ax, lab, klab in zip(a_k, l_data, axes.flatten(), t_panlab, t_klab):
    ax.text(
        0.04,
        0.96,
        lab,
        transform=ax.transAxes,
        ha="left",
        va="top",
        **get_label_style(),
    )
    ax.text(0.96, 0.96, klab, transform=ax.transAxes, ha="right", va="top")

    for tid, c in zip(a_tid, l_col):
        offset = 0 if k.item() == 0 else num_point
        (l_,) = ax.plot(
            x_ext[1:-1],
            data[3][offset:, tid],
            ls="solid",
            lw=2,
            c=c,
            alpha=0.6,
            label=f"$t = {data[2][tid].round(3)}$",
        )
        if k == a_k[0]:
            l_lines.append(l_)

    ax.margins(x=0)
    ax.set_xlabel("$x$")
    ax.set_ylim((-0.2, 6.8))

# axes settings
axes[0].set_ylabel("prob. density $p(t, x)$")
for ax in axes[1:]:
    ax.tick_params(left=False)

# put legend
fig.legend(
    handles=l_lines[: a_tid.size],
    ncols=a_tid.size,
    loc="lower center",
    bbox_to_anchor=(0.5, 1),
)

# show parameter values
d_paramstyle = put_param_area(fig)
l_nameval = [
    ("r", r),
    (r"\alpha", alpha),
    ("a", d_params["a"]),
    ("b", d_params["b"]),
    (r"\Delta x", d_params["ss_x"]),
]

fig.text(
    s=", ".join([param_val_to_str(tup[0], tup[1]) for tup in l_nameval]), **d_paramstyle
)

# show and save
if if_show == True:
    plt.show()
if if_save == True:
    fname = f"weak-pdf.pdf"
    fig.savefig(fname)

# %%
