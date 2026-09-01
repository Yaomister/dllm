import numpy as np

def calculate_entropy_k(answers, k, n_boot=1000, rng=np.random.default_rng(0)):
    """calculate the Entropy@k"""
    a = np.asarray(answers)
    n = len(a)
    if k > n:
        return np.nan
    _, codes = np.unique(a, return_inverse=True)  
    idx = np.argsort(rng.random((n_boot, n)), axis=1)[:, :k] 
    sub = codes[idx]
    counts = np.apply_along_axis(np.bincount, 1, sub, minlength=codes.max() + 1)
    p = counts / k
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.where(p > 0, p * np.log(p), 0).sum(axis=1)
    return float(h.mean())

def calculate_pass_k(n, k, c):
    """calculate the unbiased Pass@k"""

    # n = how many samples you generate
    # c = how many of the samples came out correct
    # k = how many attemps we're asking about

    # not enough failues to fill a subset, so guarenteed hit
    if n - c < k:
        return 1.0

    return 1.0 - np.prod(1 - k / np.arange(n - c + 1, n + 1))