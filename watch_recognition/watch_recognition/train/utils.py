import numpy as np


def unison_shuffled_copies(a, b, seed=42):
    """https://stackoverflow.com/a/4602224/8814045"""
    np.random.seed(seed)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
