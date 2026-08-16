import numpy as np
from collections import Counter

def calculate_entropy(answers):
    counts = np.array(list(Counter(answers).values), dtype = float)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())

def calculate_pass_k(n, k, c):
    # n = how many samples you generate
    # c = how many of the samples came out correct
    # k = how many attemps we're asking about

    # not enough failues to fill a subset, so guarenteed hit
    if n - c < k:
        return 1.0

    return 1.0 - np.prod(1 - k / np.arange(n - c + 1, n + 1))