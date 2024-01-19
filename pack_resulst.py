# pack calculation results into one npz file (on HPC)
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # uncomment when simulating on HPC

import numpy as np

from gcbistablemet.utils import get_fname, load_npz


def pack_results(N, r, alpha, a_k, num_so, per_epoch, savedir):
    if savedir != "":
        savedir = savedir + "/"

    n_trial = int(num_so * per_epoch)

    # prepare arrays
    a_fpt = np.empty(shape=(a_k.size, n_trial, N), dtype="f8")
    a_mf_fpt = np.empty(shape=(a_k.size, n_trial), dtype="f8")

    # pack calculation results
    for i, k in enumerate(a_k):
        for j in range(num_so):
            fname = get_fname(N, r, alpha, k, so=per_epoch * j)

            _d_params, _a_k, _a_fpt, _a_mf_fpt = load_npz(
                "output/" + savedir + fname, allow_pickle=True
            )

            # check simulation settings
            assert np.isclose(k, _a_k)

            _so = _d_params.pop("seed_offset")
            assert per_epoch * j == _so, "seed_offset do not match!"
            try:
                assert d_params == _d_params, "Parameter values do not match!"
            except NameError:
                d_params = dict(_d_params)
            except AssertionError as ae:
                print("others:\n", d_params)
                print(f"current (seed_offset == {_so}):\n", _d_params)
                raise ae

            a_fpt[i, per_epoch * j : per_epoch * (j + 1)] = _a_fpt
            a_mf_fpt[i, per_epoch * j : per_epoch * (j + 1)] = _a_mf_fpt

    # add seed_offset to d_params
    d_params["seed_offset"] = 0
    d_params["n_trial"] = per_epoch * num_so  # update n_trial

    # export npz file
    fname = get_fname(N, r, alpha, a_k, n_trial=n_trial)

    os.makedirs("data/" + savedir, exist_ok=True)  # make sure the directory exists
    np.savez_compressed(
        "data/" + savedir + fname,
        a_d_params=np.array([d_params], dtype="O"),
        a_k=a_k,
        a_fpt=a_fpt,
        a_mf_fpt=a_mf_fpt,
    )
    return


if __name__ == "__main__":
    args = sys.argv  # (script path, mode, ...)
    mode = args[1]

    if mode == "overall":
        N = int(args[2])
        r, alpha = (0.05, 0.1)
        a_k = np.concatenate(
            (
                25e-3 * np.arange(13, dtype="f8"),  # [0, 0.3]
                np.logspace(-1 / 2, 4, 19),
            )
        )
        per_epoch = 50
        num_so = 20
        pack_results(N, r, alpha, a_k, num_so, per_epoch, "overall")

    elif mode == "kinfty":
        N = int(args[2])
        r, alpha = (0.05, 0.1)
        a_k = np.array([10000], dtype="f8")
        per_epoch = 50
        num_so = 20
        pack_results(N, r, alpha, a_k, num_so, per_epoch, "overall")

    elif mode == "smallk":
        raise NotImplementedError
