"""Sequential visibility graph motifs (VG/HVG) of order p.

A sequential motif of order p is the subgraph induced by p consecutive data
points of a series in its visibility graph (Iacovacci & Lacasa, PRE 2016).
Since visibility between two points depends only on the *intermediate*
points, that subgraph is exactly the visibility graph of the length-p window.

Encoding
--------
Consecutive points (a, a+1) are always visible, so they carry no
information. The motif is determined by the edges between non-adjacent
pairs (a, b) with b - a >= 2. They are ordered lexicographically

    (0,2), (0,3), ..., (0,p-1), (1,3), ..., (p-3,p-1)

and the motif is encoded as an integer whose most significant bit is the
first pair. There are ``n_pairs(p) = (p-1)(p-2)/2`` bits.

For p = 3 there is a single pair (0,2) and the code is exactly the 0/1 motif
label of Table I in Iacovacci & Lacasa (2019):

* horizontal: M1 <=> x1 < x0 and x2 > x1   (x1 < min(x0, x2))
* natural:    M1 <=> x2 > 2*x1 - x0

Not every code is admissible (e.g. for order-4 HVGs only 5 of the 8 codes
can occur in tie-free data); see :func:`admissible_motifs`.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

from ._common import check_criterion

MAX_LINE_LENGTH = 12  # n_pairs(12) = 55 bits < 63: the code fits in int64


def n_pairs(length: int) -> int:
    """Number of non-adjacent pairs (code bits) in a line of ``length`` points."""
    return max(length - 1, 0) * max(length - 2, 0) // 2


@lru_cache(maxsize=None)
def motif_pairs(length: int) -> tuple[tuple[int, int], ...]:
    """Pairs (a, b), b - a >= 2, in bit order (the first one is the MSB)."""
    return tuple((a, b) for a in range(length) for b in range(a + 2, length))


def _pair_visible(lines: np.ndarray, a: int, b: int, criterion: str) -> np.ndarray:
    xa = lines[..., a]
    xb = lines[..., b]
    if criterion == "horizontal":
        # x_c < min(x_a, x_b) for every a < c < b
        return lines[..., a + 1 : b].max(axis=-1) < np.minimum(xa, xb)
    # natural: x_c < x_a + (x_b - x_a)(c - a)/(b - a); multiplied by (b - a)
    # to avoid divisions (exact for integers represented in float64).
    c = np.arange(a + 1, b)
    lhs = (b - a) * lines[..., a + 1 : b]
    rhs = (b - c) * xa[..., None] + (c - a) * xb[..., None]
    return np.all(lhs < rhs, axis=-1)


def motif_codes(lines, criterion: str = "horizontal") -> np.ndarray:
    """Sequential motif code of each line.

    Parameters
    ----------
    lines : array (..., L)
        One or more series of length L (last axis).
    criterion : {"horizontal", "natural"}

    Returns
    -------
    int64 array (...) with the motif code (see the module docstring).
    For L <= 2 the code is always 0 (trivial motif).
    """
    criterion = check_criterion(criterion)
    x = np.asarray(lines, dtype=np.float64)
    L = x.shape[-1]
    if L > MAX_LINE_LENGTH:
        raise ValueError(f"Maximum supported line length is {MAX_LINE_LENGTH} (got {L}).")
    code = np.zeros(x.shape[:-1], dtype=np.int64)
    for a, b in motif_pairs(L):
        code <<= 1
        code |= _pair_visible(x, a, b, criterion)
    return code


def code_to_edges(code: int, length: int) -> list[tuple[int, int]]:
    """Edges of the motif (including the trivial ones between consecutive points)."""
    pairs = motif_pairs(length)
    nb = len(pairs)
    edges = [(i, i + 1) for i in range(length - 1)]
    edges += [p for k, p in enumerate(pairs) if (int(code) >> (nb - 1 - k)) & 1]
    return sorted(edges)


@lru_cache(maxsize=None)
def admissible_motifs(length: int, criterion: str = "horizontal") -> tuple[int, ...]:
    """Admissible motif codes for tie-free series.

    * ``horizontal``: exact. The HVG depends only on the relative order of
      the data, so iterating over the ``length!`` permutations suffices.
      (Their number follows the Catalan numbers: 1, 1, 2, 5, 14, ...)
    * ``natural``: cannot be enumerated via permutations (it depends on the
      actual values); only implemented up to ``length = 3`` (codes {0, 1}).
    """
    criterion = check_criterion(criterion)
    if length <= 2:
        return (0,)
    if criterion == "natural":
        if length == 3:
            return (0, 1)
        raise NotImplementedError(
            "Exact enumeration of natural VG motifs is only implemented for length <= 3."
        )
    if length > 9:
        raise ValueError("Enumeration via permutations is limited to length <= 9.")
    perms = np.array(list(itertools.permutations(range(length))), dtype=np.float64)
    return tuple(sorted(set(motif_codes(perms, "horizontal").tolist())))
