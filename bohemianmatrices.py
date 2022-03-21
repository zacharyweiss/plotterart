"""Experiments with 'Bohemian' matrices / eigenvalues, as prompted by http://www.bohemianmatrices.com/"""

import numpy as np
import colorcet as cc
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from fast_histogram import histogram2d
from math import log10
import datetime


def samples(A: np.array, n: int, bounds: tuple, entries: list[tuple], sample_coef=None, sample_offset=None) -> np.array:
    assert type(bounds) is tuple and len(bounds) == 2
    for entry in entries:
        assert type(entry) is tuple and len(entry) == 2

    if sample_coef is None:
        sample_coef = 1

    if sample_offset is None:
        sample_offset = 0

    A = A.astype('cdouble')
    A = np.repeat(A[None, ...], int(n), axis=0)

    for entry in entries:
        rvs = sample_coef * np.random.uniform(low=bounds[0], high=bounds[1], size=int(n)) + sample_offset
        A[:, entry[0], entry[1]] = rvs

    return A


def samples_auto_entries(A: np.array, n: int, bounds: tuple, **kwargs) -> np.array:
    empty = np.where(A == None)
    entries = list(zip(empty[0], empty[1]))
    return samples(A, n, bounds, entries, **kwargs)


def to_eigs(A: np.array, return_components: bool = False, exclude_real: bool = True):
    assert A.ndim == 3

    eigs = set(np.linalg.eigvals(A).flatten())

    if return_components and exclude_real:
        return [x.real for x in eigs if x.imag != 0], [x.imag for x in eigs if x.imag != 0]
    elif return_components and not exclude_real:
        return [x.real for x in eigs], [x.imag for x in eigs]
    else:
        return eigs

def generate_many(n_sample=5E5, n_image=1, max_uniform=5, range_dim=(3, 6), range_var=(1, 4),
                  range_coef_re=(-9, 10), range_coef_im=(-9, 10), range_offs_re=(-9, 10), range_offs_im=(-9, 10),
                  suffix: str = ""):
    cmap = cc.cm["gray"].copy()
    cmap.set_bad(cmap.get_under())

    r = lambda l, h=None: np.random.randint(l, h)
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d")
    for i in range(n_image):
        n_dim = r(range_dim[0], range_dim[1])
        A = np.random.randint(-1, 2, size=(n_dim, n_dim))
        A = A.astype('object')

        for j in range(r(range_var[0], range_var[1])):
            A[r(n_dim), r(n_dim)] = None
        A_str = f"{A}"

        if r(2):
            coef = None
            offs = None
            # some uniform distribution
            rand_max = r(1, max_uniform)
            bounds = (-rand_max, rand_max)
            A = samples_auto_entries(A, int(n_sample), bounds)
        else:
            #
            coef = r(range_coef_re[0], range_coef_re[1]) + 1j * r(range_coef_im[0], range_coef_im[1])
            offs = r(range_offs_re[0], range_offs_re[1]) + 1j * r(range_offs_im[0], range_offs_im[1])
            bounds = (0, 1)
            A = samples_auto_entries(A, int(n_sample), bounds, sample_coef=coef, sample_offset=offs)

        x, y = to_eigs(A, return_components=True)
        if not x or len(set(x)) < 0.1*n_sample:
            continue  # skip empty

        h = histogram2d(y, x, range=[[min(y), max(y)], [min(x), max(x)]], bins=int(np.sqrt(2 * n_sample)))
        fig, ax = plt.subplots()
        ax.imshow(h, norm=colors.LogNorm(vmin=1, vmax=h.max()), cmap=cmap)
        plt.axis('off')
        plt.show()

        print(f"\n#{i}")
        print(f"{A_str}")
        print(f"Coef: {coef}")
        print(f"Offs: {offs}")
        print(f"Bounds: {bounds}")
        metadata = {
            'Author': 'Zachary Weiss',
            'Description': f"""Matrix:
{A_str}
Coef: {coef}
Offs: {offs}
Bounds: {bounds}""",
        }

        fn = f'{now_str}_{i:0{int(log10(n_image))}d}{suffix}.png'
        fig.savefig(fn, bbox_inches="tight", pad_inches=0,
                    dpi=int(np.sqrt(2 * n_sample)),
                    metadata=metadata)


def export(A: np.array, export_name: str, n_sample: int = 1E7, bounds=(0, 1), sample_coef=None, sample_offset=None):
    A = samples_auto_entries(A, int(n_sample), bounds, sample_coef=sample_coef, sample_offset=sample_offset)
    x, y = to_eigs(A, return_components=True)

    # cmap = cc.cm["fire"].copy()
    cmap = cc.cm["gray"].copy()
    cmap.set_bad(cmap.get_under())

    # for some reason the fast histogram gives the transposed image, hence switching x & y in args for corrected output
    h = histogram2d(y, x, range=[[min(y), max(y)], [min(x), max(x)]], bins=int(np.sqrt(2 * n_sample)))
    fig, ax = plt.subplots()
    ax.imshow(h, norm=colors.LogNorm(vmin=1, vmax=h.max()), cmap=cmap)
    plt.axis('off')
    # fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # fig.tight_layout(pad=0)

    plt.show()

    fig.savefig(export_name, bbox_inches="tight", pad_inches=0, dpi=int(np.sqrt(2 * n_sample)))
    print(f"{export_name} saved.")


def main():
    # generate_many(n_image=100, suffix='-1')
    """export(np.array([[0, -1, 0, 0], [None, None, None, 1], [-1, 1, 1, 1], [-1, 0, -1, 1]]),
           "test4.png", sample_coef=8-7j, sample_offset=-2+2j)
    export(np.array([[None, 1, 0], [-1, 0, None], [-1, 0, None]]),
           "test5.png", sample_coef=4-6j, sample_offset=-2)
    export(np.array([[-1, 1, None, 1], [1, 1, -1, 0], [None, 0, None, 1], [0, 0, -1, 1]]),
           "test6.png", bounds=(-4, 4))
    export(np.array([[-1, 0, None], [1, 0, 1], [None, None, -1]]),
           "test7.png", sample_coef=-8+8j, sample_offset=1)
    export(np.array([[1, None, 1], [1, None, None], [0, -1, -1]]),
           "test9.png", sample_coef=9-9j, sample_offset=-1)
    export(np.array([[None, 1, 0, 0], [0, None, 1, 1], [1, -1, -1, 1], [-1, -1, 0, 1]]),
           "test10.png", bounds=(-3, 3))
    export(np.array([[-1, -1, 1, 1, None], [1, 0, 0, 0, 0], [-1, -1, None, -1, -1], [-1, -1, -1, 0, 1], [0, 1, 1, None, 1]]),
           "test11.png", bounds=(-3, 3))
    export(np.array([[0, None, -1, 0], [-1, 1, None, 0], [0, -1, 1, 1], [1, -1, 1, 1]]),
           "test12.png", sample_coef=5-6j, sample_offset=-2+4j)
    export(np.array([[1, None, 0, 1], [0, None, 1, -1], [0, -1, -1, 1], [None, 0, -1, 1]]),
           "test13.png", sample_coef=4-7j, sample_offset=-2+6j)
    export(np.array([[1, None, 0], [0, None, -1], [1, 0, 1]]),
           "test14.png", bounds=(-4, 4))
    export(np.array([[1, 0, 0, 0], [None, None, 0, -1], [0, 0, 0, None], [-1, -1, 1, 0]]),
           "test15.png", sample_coef=-3-7j, sample_offset=2+3j)
    export(np.array([[None, None, 1], [None, -1, 1], [0, 0, -1]]),
           "test16.png", sample_coef=4+6j, sample_offset=-4-3j)"""


if __name__ == "__main__":
    np.random.seed(5)
    main()
