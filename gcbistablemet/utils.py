import numpy as np


def load_npz(fname: str, allow_pickle=False):
    """
    Unpack a npz file.
    If the file contains an array named "a_d_params", its first element will be returned.
    """
    with np.load(fname, allow_pickle=allow_pickle) as data:
        l_arr = []
        for key in data.files:
            if key == "a_d_params":
                d_params = data["a_d_params"][0]
            else:
                l_arr.append(data[key])
        try:
            return d_params, *l_arr
        except NameError:
            return l_arr


def get_fname(
    N: int, r: float, alpha: float, a_k: np.ndarray, so: int = None, n_trial: int = None
):
    if a_k.size == 1:
        f_k: float = a_k.round(10).item()
        k_str: str = f"k{int(f_k):05}_{int(10000*(f_k-int(f_k))):04}"
    else:
        k_str: str = f"nk{a_k.size}"

    fname: str = f"n{N}-r{round(100*r):03}-alpha{round(100*alpha):03}-{k_str}-"
    if n_trial is None:
        fname += f"so{so:04}.npz"
    else:
        fname += f"nt{n_trial}.npz"

    return fname
