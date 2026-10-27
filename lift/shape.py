"""Classify an element-level dependency relation R (n_out x n_in) into a
combinator shape.  Purely deterministic."""
import itertools
import numpy as np


def grid_candidates(n, ints):
    c = []
    for a, b in itertools.permutations(sorted(set(int(i) for i in ints if int(i) > 1)), 2):
        if a * b == n:
            c.append((a, b))
    return c


def classify(R, ints=()):
    no, ni = R.shape
    if no == 1:
        return ('fold', {'over': ni}) if R.all() else ('partial-fold', {'nnz': int(R.sum())})
    if no == ni and np.array_equal(R, np.eye(ni, dtype=bool)):
        return ('map', {})
    if no == ni:
        for nx, ny in grid_candidates(ni, ints):
            S, deps = set(), []
            for k in range(no):
                i, j = k % nx, k // nx
                d = {(jj % nx - i, jj // nx - j) for jj in np.flatnonzero(R[k])}
                deps.append((i, j, d)); S |= d
            ok = all(d == {(di, dj) for di, dj in S if 0 <= i + di < nx and 0 <= j + dj < ny}
                     for i, j, d in deps)
            if ok:
                return ('stencil2d', {'grid': (nx, ny), 'offsets': sorted(S),
                                      'boundary': 'truncated (homogeneous Dirichlet)'})
    return ('general', {'nnz': int(R.sum())})
