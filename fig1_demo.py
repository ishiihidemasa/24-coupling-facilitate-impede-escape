# %%[markdown]
# # Illustration of deterministic / stochastic bistable systems

# %%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp

import gcbistablemet
from gcbistablemet.gc_bistable import schloegl
from gcbistablemet.integrate import euler_maruyama
from gcbistablemet.plot_utils import get_label_style, param_val_to_str, put_param_area


# %%
def schloegl_pot(x, r):
    return np.power(x, 4) / 4 - (1 + r) / 3 * np.power(x, 3) + r / 2 * x * x


# %%
# flags
if_show: bool = True  # If False, plt.show() won't be called.
if_save: bool = False  # If False, fig.savefig() won't be called.

plt.style.use("gcbistablemet.paperfigure")

# %%
r = 0.35
init = np.array([-0.2, 0.25, 1.2], dtype="f8")

# %%
# ODE
res_det = solve_ivp(schloegl, (0, 20), init, args=(r,), atol=1e-8, rtol=1e-8)

# %%
# SDE
alpha = 0.15
t_max = 300
dt = 1e-2
ss_t = 0.5
seed = 0

a_time = dt * np.arange(round(t_max / dt) + 1, dtype="f8")

res_sto = euler_maruyama(
    schloegl,
    (r,),
    lambda t, x: alpha,
    (),
    a_time,
    init,
    a_time[:: round(ss_t / dt)],
    seed=seed,
)

# %%
fig = plt.figure(figsize=(7, 3.4))
# create subplots
gs = GridSpec(nrows=2, ncols=2, figure=fig, width_ratios=(2, 3))
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax0 = fig.add_subplot(gs[:, 0])
ax2.sharey(ax1)

# put panel labels and descriptions
t_panlab = ("a", "b", "c")
t_desc = ("potential function", "deterministic trajectories", "stochastic trajectories")

ax0.text(
    0.03,
    0.94,
    t_panlab[0],
    transform=ax0.transAxes,
    ha="left",
    va="center",
    **get_label_style(),
)
ax0.text(0.12, 0.94, t_desc[0], transform=ax0.transAxes, ha="left", va="center")

for ax, panlab, desc in zip((ax1, ax2), t_panlab[1:], t_desc[1:]):
    ax.text(
        0.98,
        0.02,
        panlab,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        **get_label_style(),
    )

    ax.set_title(desc, loc="left", size="medium", pad=4)

## panel a
col, m = ("k", "|")
pad = 2e-3
x = np.linspace(-0.2, 1.27, 101)
ax0.plot(x, schloegl_pot(x, r), c=col, alpha=0.8, lw=2)
vl = schloegl_pot(0, r)
vm = schloegl_pot(r, r)
vr = schloegl_pot(1, r)
ax0.plot(0, vl, c=col, ls="none", marker=m)
ax0.plot(r, vm, c=col, ls="none", marker=m)
ax0.plot(1, vr, c=col, ls="none", marker=m)
ax0.text(0, vl - pad, "lower state", ha="center", va="top")
ax0.text(r, vm + pad, "$x = r$", ha="center", va="bottom")
ax0.text(1, vr - pad, "upper state", ha="center", va="top")

ax0.set_xlabel("state $x$")
ax0.margins(x=0.1)
ax0.ticklabel_format(axis="y", scilimits=(-1, 1), useMathText=True)
ax0.set_ylim((vr * 1.25, 2e-2))

## panels b & c
for ax in (ax1, ax2):
    # horizontal lines at attracting states
    col = "tab:gray"
    ax.axhline(0, c=col, lw=1, zorder=1)
    ax.axhline(1, c=col, lw=1, zorder=1)
    ax.axhline(r, c=col, lw=1, ls="--", zorder=1)
    # axes labels and plot range
    ax.set_ylabel("state $x$")
    ax.margins(x=0)
    ax.set_ylim((-0.55, 1.35))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


ax1.plot(res_det.t, res_det.y.T, alpha=0.7, lw=2)

ax2.plot(res_sto.t, res_sto.y.T, alpha=0.7, lw=1)
ax2.set_xlabel("time $t$")

# show parameter values
d_paramstyle = put_param_area(fig)
l_nameval = [("r", r), (r"\alpha", alpha)]
l_nameval_sto = [
    (r"\mathrm{dt}", dt),
    (r"\Delta t", ss_t),
    (r"\mathrm{seed}", seed),
]

concat = lambda l_tup: ", ".join([param_val_to_str(tup[0], tup[1]) for tup in l_tup])
s = concat(l_nameval)
s += r", $x(t=0) \in \{" + ", ".join([f"{i_}" for i_ in init]) + "\}$"
s += "; panel (c): " + concat(l_nameval_sto)

fig.text(s=s, **d_paramstyle)

# show & save
if if_show == True:
    plt.show()
if if_save == True:
    fig.savefig("demo.pdf")

# %%
