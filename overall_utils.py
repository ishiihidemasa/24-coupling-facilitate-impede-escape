# Load npz files for N = 50, 100, and 200
import numpy as np

from gcbistablemet.utils import load_npz


def load_overall(a_N):
    savedir = "overall/"
    fn = lambda N: f"n{N}-r005-alpha010-nk32-nt2000.npz"

    l_data = [load_npz("data/" + savedir + fn(N_), allow_pickle=True) for N_ in a_N]

    for n_, data in zip(a_N, l_data):
        d_ = dict(data[0])
        n_in_dict = d_.pop("N")
        assert n_ == n_in_dict, "N do not match!"

        try:
            assert np.isclose(a_k, data[1]).all(), "a_k do not match!"
        except NameError:
            a_k = data[1]

        try:
            assert d_ == d_params
        except NameError:
            d_params = dict(d_)
        except AssertionError as ae:
            print("others:\n", d_params)
            print(f"current (N == {n_}):\n", d_)
            raise ae

    return d_params, a_k, l_data


def get_markers() -> tuple[str]:
    return ("o", "^", "s")


def get_colors() -> tuple[str]:
    return ("tab:blue", "tab:green", "tab:orange")
