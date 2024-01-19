# %%[markdown]
# # Figure 5
# Compare mean and SD of average escape times

# %%
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import gcbistablemet
from gcbistablemet.plot_utils import get_label_style, param_val_to_str, put_param_area
from gcbistablemet.utils import get_fname, load_npz
from gcbistablemet.weak_coupling_theory import prob_current

# %%
# flags
if_show: bool = True  # If False, plt.show() won't be called.
if_save: bool = False  # If False, fig.savefig() won't be called.

# %%
# common settings
plt.style.use("gcbistablemet.paperfigure")

r: float = 0.05
a_alpha = np.array([0.02, 0.1], dtype="f8")
a_k_002 = 1e-2 * np.arange(7, dtype="f8")
a_k_01 = 5e-2 * np.arange(7, dtype="f8")
N: int = 200

# %%
# load simulation data
l_data_sim = []
for alpha, a_k in zip(a_alpha, (a_k_002, a_k_01)):
    fname = get_fname(N, r, alpha, a_k, n_trial=1000)
    data = load_npz("data/smallk/" + fname, allow_pickle=True)
    d_ = data[0]
    assert np.isclose(alpha, d_.pop("alpha"))
    try:
        assert d_ == d_params_sim, "Different settings!"
    except NameError:
        d_params_sim = d_
    except AssertionError as ae:
        print("current:\n", d_)
        print("others:\n", d_params_sim)
        raise ae
    l_data_sim.append(data)

xi = d_params_sim["xi"]

# %%
# load theory data
l_data_theory = []
l_d_params_t = []
for alpha, a_k in zip(a_alpha, (a_k_002, a_k_01)):
    l_data_ = []

    try:
        del d_params_theory
    except NameError:
        pass

    for k in tqdm(a_k):
        fname = f"r{round(r*1e2):03}-alpha{round(alpha*1e2):03}-k{round(k.item()*1e3):04}.npz"
        data = load_npz("data/weaktheory/" + fname, allow_pickle=True)
        d_ = dict(data[0])
        assert np.isclose(k, d_.pop("k"))
        assert np.isclose(alpha, d_.pop("alpha"))
        try:
            assert d_ == d_params_theory, "Different settings!"
        except NameError:
            d_params_theory = d_
        except AssertionError as ae:
            print("current:\n", d_)
            print("others:\n", d_params_theory)
            raise ae
        l_data_.append(data)

    l_data_theory.append(l_data_)
    l_d_params_t.append(d_params_theory)

# %%
# create figure
fig, axes = plt.subplots(figsize=(7, 2.4), ncols=2)

# axes settings
for ax in axes:
    ax.set_xlabel("coupling strength $K$")
axes[0].set_ylabel(r"escape time $\tau_i$")

# put panel labels and descriptions
t_panlab = ("a", "b")
for ax, panlab, alpha in zip(axes, t_panlab, a_alpha):
    ax.text(
        0.96,
        0.96,
        panlab,
        ha="right",
        va="top",
        transform=ax.transAxes,
        **get_label_style(),
    )
    ax.text(
        0.96, 0.82, rf"$\alpha = {alpha}$", ha="right", va="top", transform=ax.transAxes
    )

# show results
for ax, alpha, a_k, data_sim, data_theory, d_params_theory in zip(
    axes, a_alpha, (a_k_002, a_k_01), l_data_sim, l_data_theory, l_d_params_t
):
    # direct simulation
    a_fpt = data_sim[2]

    l_s = ax.errorbar(
        a_k,
        a_fpt.mean(axis=2).mean(axis=1),
        yerr=a_fpt.std(axis=2).mean(axis=1),
        marker="o",
        ms=7,
        mfc="none",
        mew=2,
        ls="none",
        capsize=3,
        label="simulation",
        zorder=5,
    )

    # theory
    a_met = np.empty(a_k.size, dtype="f8")
    a_sd = np.empty(a_k.size, dtype="f8")

    for i, (k, data) in enumerate(zip(a_k, data_theory)):
        res_t, res_y = (data[2], data[3])

        # probability current: dim == (position, time)
        current = prob_current(
            res_y,
            r,
            alpha,
            k,
            xi,
            data[1],
            d_params_theory["a"],
            d_params_theory["ss_x"],
        )

        # moments of escape times
        m1 = np.sum(res_t[:-1] * current[1, :-1] * (res_t[1:] - res_t[:-1]))
        m2 = np.sum(
            np.power(res_t[:-1], 2) * current[1, :-1] * (res_t[1:] - res_t[:-1])
        )
        sd = np.sqrt(m2 - m1 * m1)

        a_met[i] = m1
        a_sd[i] = sd

    c_fpe = "tab:orange"
    ax.fill_between(
        a_k, y1=a_met + a_sd, y2=a_met - a_sd, color=c_fpe, alpha=0.2, zorder=1
    )
    (l_t,) = ax.plot(a_k, a_met, ls=":", c=c_fpe, label="theory", zorder=2)

# put legend
fig.legend(handles=[l_s, l_t], ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1))

# put parameter values
d_paramstyle = put_param_area(fig)
l_nameval = [
    ("r", r),
    (r"\xi", xi),
]
l_nameval_sim = [
    ("N", N),
    (r"\mathrm{dt}", d_params_sim["dt"]),
    (r"\Delta t", d_params_sim["ss_t"]),
    (r"n_{\, \mathrm{trial}}", d_params_sim["n_trial"]),
]
l_nameval_t = [
    (r"\Delta x", d_params_theory["ss_x"]),
    (r"\Delta t", d_params_theory["ss_t"]),
    (r"X_{\mathrm{end}}", d_params_theory["mf_end"]),
]
l_nameval_t_a = [
    ("a", l_d_params_t[0]["a"]),
    ("b", l_d_params_t[0]["b"]),
]
l_nameval_t_b = [
    ("a", l_d_params_t[1]["a"]),
    ("b", l_d_params_t[1]["b"]),
]

concat = lambda l_tup: ", ".join([param_val_to_str(tup[0], tup[1]) for tup in l_tup])
s_common: str = concat(l_nameval)
s_sim: str = concat(l_nameval_sim)
s_t: str = concat(l_nameval_t)
s_t_a: str = concat(l_nameval_t_a)
s_t_b: str = concat(l_nameval_t_b)
s: str = s_common + "; simulation: " + s_sim + "\n"
s += "theory: " + s_t + "; panel (a): " + s_t_a + "; panel (b): " + s_t_b

fig.text(s=s, **d_paramstyle)

# show & save
if if_show == True:
    plt.show()
if if_save == True:
    figname = "weak-met-sd-comp.pdf"
    fig.savefig(figname)

# %%
