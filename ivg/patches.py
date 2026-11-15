"""Visibility Patches (VP_p / HVP_p) and the patch profile Z.

Reference: Iacovacci, J. & Lacasa, L. (2019). Visibility graphs for image
processing. IEEE TPAMI 42(4), 974-987. Section II.C.3.

Definition
----------
A Visibility Patch of order p is the subgraph of the IVG (natural) or the
IHVG (horizontal) induced by the p×p pixels of a window. Within the window,
the IVG only connects pixels lying on the same row, column, diagonal or
anti-diagonal, and visibility between two pixels depends only on the
intermediate pixels (which also lie inside the window). Hence the patch is
fully determined by the sequential motifs of each line of the window:

* p rows and p columns (order-p motifs),
* main diagonal and anti-diagonal (order p),
* q = 4(p-2) off-diagonals: 4 of order p-1, 4 of order p-2, ..., 4 of order 2.

Line order (the vector V of the paper):

    [r1..rp, c1..cp, d1, d2, d1+1, d1-1, d2+1, d2-1, d1+2, d1-2, d2+2, d2-2, ...]

where ``d1±o`` is the diagonal shifted o positions above (+) or below (-)
the main diagonal, and ``d2±o`` the same for the anti-diagonal. Length-2
diagonals are trivial (0 bits) but are listed for completeness.

Label
-----
Each line contributes the code of its motif (see :mod:`ivg.motifs`) with
``n_pairs(L)`` bits. The patch label is the concatenation of those bits
(first line most significant) plus 1:

    L = 1 + concat_bits(V)

For p = 3 this reproduces the paper's encoding exactly:
L = 1 + sum_i V_i 2^(8-i), with 256 possible labels (Fig. 4: [1,0,0,0,0,1,1,0] -> 135).
For p >= 4 the label space is huge (2^34 for p = 4) but very sparse, so the
profile is stored sparsely (only observed labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ._common import as_image, check_criterion
from .motifs import MAX_LINE_LENGTH, code_to_edges, motif_codes, n_pairs

DENSE_MAX_BITS = 24  # to_dense() only if the label space is <= 2^24
_CHUNK_ELEMS = 2_000_000  # approx. number of float64 values processed per block


class PatchLine(NamedTuple):
    """One line (row, column or diagonal) of a p×p patch."""

    name: str
    coords: np.ndarray  # (L, 2) (row, col) indices within the patch, in order

    @property
    def length(self) -> int:
        return len(self.coords)

    @property
    def n_bits(self) -> int:
        return n_pairs(self.length)


@lru_cache(maxsize=None)
def patch_layout(order: int) -> tuple[PatchLine, ...]:
    """Lines of an order-p patch, in the order of the vector V."""
    p = _check_order(order)
    k = np.arange(p)
    lines = [PatchLine(f"r{r + 1}", np.stack([np.full(p, r), k], 1)) for r in range(p)]
    lines += [PatchLine(f"c{c + 1}", np.stack([k, np.full(p, c)], 1)) for c in range(p)]
    lines.append(PatchLine("d1", np.stack([k, k], 1)))
    # Anti-diagonal in the paper's direction: (i, j+p-1) -> (i+p-1, j)
    lines.append(PatchLine("d2", np.stack([k, p - 1 - k], 1)))
    for o in range(1, p - 1):
        m = np.arange(p - o)
        lines.append(PatchLine(f"d1+{o}", np.stack([m, m + o], 1)))
        lines.append(PatchLine(f"d1-{o}", np.stack([m + o, m], 1)))
        lines.append(PatchLine(f"d2+{o}", np.stack([m, p - 1 - o - m], 1)))
        lines.append(PatchLine(f"d2-{o}", np.stack([m + o, p - 1 - m], 1)))
    for ln in lines:
        ln.coords.setflags(write=False)
    return tuple(lines)


def label_bits(order: int) -> int:
    """Total number of bits in an order-p patch label (8 for p=3, 34 for p=4)."""
    return sum(ln.n_bits for ln in patch_layout(order))


def n_labels(order: int) -> int:
    """Size of the label space (256 for p = 3)."""
    return 1 << label_bits(order)


def _check_order(order) -> int:
    p = int(order)
    if p != order or p < 3:
        raise ValueError(f"order must be an integer >= 3 (got {order!r}).")
    if p > MAX_LINE_LENGTH:
        raise ValueError(f"Maximum supported order is {MAX_LINE_LENGTH}.")
    return p


def _code_dtype(order: int):
    nb = n_pairs(order)
    if nb <= 8:
        return np.uint8
    if nb <= 16:
        return np.uint16
    if nb <= 32:
        return np.uint32
    return np.int64


def _check_stride(stride) -> int:
    s = int(stride)
    if s != stride or s < 1:
        raise ValueError(f"stride must be an integer >= 1 (got {stride!r}).")
    return s


def patch_codes(img, order: int = 3, criterion: str = "horizontal", stride: int = 1) -> np.ndarray:
    """Motif code of every line of every patch.

    Returns an array ``(H', W', n_lines)``; patch ``[i, j]`` is
    ``img[i*stride : i*stride+p, j*stride : j*stride+p]`` and lines follow
    :func:`patch_layout`. With the default ``stride=1`` (as in the paper),
    ``H' = H-p+1`` and ``W' = W-p+1``.
    """
    p = _check_order(order)
    criterion = check_criterion(criterion)
    stride = _check_stride(stride)
    I = as_image(img, min_size=p)
    windows = sliding_window_view(I, (p, p))[::stride, ::stride]  # view without copy
    Hp, Wp = windows.shape[:2]
    layout = patch_layout(p)
    out = np.empty((Hp, Wp, len(layout)), dtype=_code_dtype(p))

    # Group lines by length to evaluate them in bulk
    by_len: dict[int, list[int]] = {}
    for idx, ln in enumerate(layout):
        by_len.setdefault(ln.length, []).append(idx)

    for L, idxs in by_len.items():
        if L <= 2:
            out[..., idxs] = 0
            continue
        R = np.stack([layout[i].coords[:, 0] for i in idxs])  # (n, L)
        C = np.stack([layout[i].coords[:, 1] for i in idxs])
        rows_per_chunk = max(1, _CHUNK_ELEMS // max(1, Wp * len(idxs) * L))
        for s in range(0, Hp, rows_per_chunk):
            block = windows[s : s + rows_per_chunk][..., R, C]  # (h, W', n, L)
            out[s : s + rows_per_chunk, :, idxs] = motif_codes(block, criterion)
    return out


def _labels_from_codes_int64(codes: np.ndarray, order: int) -> np.ndarray:
    lab = np.zeros(codes.shape[:-1], dtype=np.int64)
    for i, ln in enumerate(patch_layout(order)):
        if ln.n_bits:
            lab <<= ln.n_bits
            lab |= codes[..., i].astype(np.int64)
    return lab + 1


def _label_from_code_row(row: Sequence[int], order: int) -> int:
    lab = 0
    for c, ln in zip(row, patch_layout(order)):
        lab = (lab << ln.n_bits) | int(c)
    return lab + 1


def patch_label_map(img, order: int = 3, criterion: str = "horizontal", stride: int = 1) -> np.ndarray:
    """Map with the label (>= 1) of every patch (shape ``(H-p+1, W-p+1)`` for stride 1).

    Only available when the label fits in int64 (p <= 4). For higher orders
    use :func:`patch_profile` and its ``label_map()`` method.
    """
    p = _check_order(order)
    if label_bits(p) > 62:
        raise ValueError(
            f"For order={p} the label has {label_bits(p)} bits and does not fit in int64; "
            "use patch_profile(...).label_map() instead."
        )
    return _labels_from_codes_int64(patch_codes(img, p, criterion, stride), p)


# Order-3 labels in the reference MATLAB code (Jacopo Iacovacci, github.com/Jaia89/Image_Visibility)
# use a different numbering from Algorithm 1 of the paper: the bit string is
# [r1, c3, r3, c1, c2, r2, d1, d2] and its first element is the LEAST significant bit.
# The set of patches is identical; only the numbering differs.
_MATLAB_LINE_ORDER = ("r1", "c3", "r3", "c1", "c2", "r2", "d1", "d2")
_PAPER_LINE_ORDER = ("r1", "r2", "r3", "c1", "c2", "c3", "d1", "d2")


@lru_cache(maxsize=None)
def _paper_to_matlab_table() -> np.ndarray:
    table = np.zeros(257, dtype=np.int64)
    for lab in range(1, 257):
        v = lab - 1
        bits = {name: (v >> (7 - i)) & 1 for i, name in enumerate(_PAPER_LINE_ORDER)}
        table[lab] = 1 + sum(bits[name] << k for k, name in enumerate(_MATLAB_LINE_ORDER))
    table.setflags(write=False)
    return table


def paper_to_matlab_label(label):
    """Convert order-3 labels from the paper's numbering to the reference MATLAB numbering.

    Use it to compare results with ``visibilityPatches.m``. Accepts an int or an array.
    """
    lab = np.asarray(label, dtype=np.int64)
    if np.any((lab < 1) | (lab > 256)):
        raise ValueError("Order-3 labels must lie in [1, 256].")
    out = _paper_to_matlab_table()[lab]
    return int(out) if out.ndim == 0 else out


def matlab_to_paper_label(label):
    """Inverse of :func:`paper_to_matlab_label`."""
    lab = np.asarray(label, dtype=np.int64)
    if np.any((lab < 1) | (lab > 256)):
        raise ValueError("Order-3 labels must lie in [1, 256].")
    inv = np.zeros(257, dtype=np.int64)
    inv[_paper_to_matlab_table()[1:]] = np.arange(1, 257)
    out = inv[lab]
    return int(out) if out.ndim == 0 else out


def decode_label(label: int, order: int = 3) -> dict:
    """Decode a label into the motif code of each line.

    Returns a dict ``{line_name: code}``; for p = 3 the codes of the 8 main
    lines are exactly the bits V(1..8) of the paper. Use
    :func:`ivg.motifs.code_to_edges` to see the edges of each motif.
    """
    p = _check_order(order)
    v = int(label) - 1
    if not 0 <= v < n_labels(p):
        raise ValueError(f"Label out of range [1, {n_labels(p)}] for order={p}.")
    out = {}
    for ln in reversed(patch_layout(p)):
        out[ln.name] = v & ((1 << ln.n_bits) - 1)
        v >>= ln.n_bits
    return {ln.name: out[ln.name] for ln in patch_layout(p)}


def decode_label_edges(label: int, order: int = 3) -> dict:
    """Like :func:`decode_label`, but returning the edges (in patch coordinates) of each line."""
    p = _check_order(order)
    res = {}
    for ln, (name, code) in zip(patch_layout(p), decode_label(label, p).items()):
        res[name] = [
            (tuple(ln.coords[a]), tuple(ln.coords[b])) for a, b in code_to_edges(code, ln.length)
        ]
    return res


@dataclass
class PatchProfile:
    """Visibility Patch profile of an image (sparse representation).

    Attributes
    ----------
    order, criterion : parameters used.
    labels : array (n,) of Python ints (object dtype) with the observed labels, sorted.
    counts : array (n,) with the number of patches carrying each label.
    codes  : array (n, n_lines) with the per-line motif code of each label.
    shape  : shape of the patch grid ((H-p+1, W-p+1) for stride 1).
    inverse : array with that shape holding the index into ``labels`` of each patch.
    """

    order: int
    criterion: str
    labels: np.ndarray
    counts: np.ndarray
    codes: np.ndarray
    shape: tuple[int, int]
    inverse: np.ndarray

    @property
    def n_patches(self) -> int:
        return int(self.counts.sum())

    def __len__(self) -> int:
        return len(self.labels)

    def frequencies(self) -> np.ndarray:
        """Relative frequencies (summing to 1), as the paper's profile Z."""
        return self.counts / self.counts.sum()

    def as_dict(self, normalize: bool = False) -> dict[int, float]:
        vals = self.frequencies() if normalize else self.counts
        return {int(lab): v.item() for lab, v in zip(self.labels, vals)}

    def to_series(self, normalize: bool = False):
        """``pandas.Series`` indexed by label."""
        import pandas as pd

        vals = self.frequencies() if normalize else self.counts
        return pd.Series(vals, index=pd.Index(self.labels, name="label"), name="Z")

    def most_common(self, n: int = 10) -> list[tuple[int, int]]:
        order = np.argsort(-self.counts, kind="stable")[:n]
        return [(int(self.labels[i]), int(self.counts[i])) for i in order]

    def to_dense(self, normalize: bool = False, numbering: str = "paper") -> np.ndarray:
        """Dense vector of length ``n_labels(order)``; position ``l-1`` holds label ``l``.

        For p = 3 this is the paper's Z = (Z_1, ..., Z_256). With
        ``numbering="matlab"`` (p = 3 only) the entries follow the numbering of
        the reference MATLAB code, so the result can be compared directly
        with the output of ``visibilityPatches.m``.
        """
        if label_bits(self.order) > DENSE_MAX_BITS:
            raise ValueError(
                f"For order={self.order} there are 2^{label_bits(self.order)} possible labels; "
                "use as_dict()/to_series(), or profile_matrix() for several images."
            )
        if numbering not in ("paper", "matlab"):
            raise ValueError("numbering must be 'paper' or 'matlab'.")
        labels = np.asarray(self.labels, dtype=np.int64)
        if numbering == "matlab":
            if self.order != 3:
                raise ValueError("numbering='matlab' is only defined for order 3.")
            labels = paper_to_matlab_label(labels)
        vals = self.frequencies() if normalize else self.counts
        Z = np.zeros(n_labels(self.order), dtype=vals.dtype)
        Z[labels - 1] = vals
        return Z

    def label_map(self) -> np.ndarray:
        """Per-patch label map (int64 when it fits; otherwise an object array of Python ints)."""
        if label_bits(self.order) <= 62:
            return np.asarray(self.labels, dtype=np.int64)[self.inverse]
        return self.labels[self.inverse]


def patch_profile(
    img, order: int = 3, criterion: str = "horizontal", stride: int = 1
) -> PatchProfile:
    """Order-p Visibility Patch profile (HVP_p if horizontal, VP_p if natural).

    ``stride`` is the step between consecutive patches (1 in the paper).

    Example
    -------
    >>> prof = patch_profile(img, order=3, criterion="horizontal")
    >>> Z = prof.to_dense(normalize=True)   # vector of 256 relative frequencies
    """
    p = _check_order(order)
    criterion = check_criterion(criterion)
    codes = patch_codes(img, p, criterion, stride)
    Hp, Wp, n = codes.shape
    flat = np.ascontiguousarray(codes.reshape(-1, n))
    rows = flat.view(np.dtype((np.void, flat.dtype.itemsize * n))).ravel()
    _, first, inverse, counts = np.unique(
        rows, return_index=True, return_inverse=True, return_counts=True
    )
    ucodes = flat[first]
    labels = np.array([_label_from_code_row(r, p) for r in ucodes.tolist()], dtype=object)
    perm = sorted(range(len(labels)), key=lambda i: labels[i])
    perm = np.asarray(perm, dtype=np.int64)
    rank = np.empty_like(perm)
    rank[perm] = np.arange(len(perm))
    return PatchProfile(
        order=p,
        criterion=criterion,
        labels=labels[perm],
        counts=counts[perm].astype(np.int64),
        codes=ucodes[perm],
        shape=(Hp, Wp),
        inverse=rank[inverse.ravel()].reshape(Hp, Wp),
    )


def profile_matrix(profiles: Sequence[PatchProfile], normalize: bool = True, labels=None):
    """Feature matrix (n_images × n_labels) built from several profiles.

    Columns are the sorted union of observed labels (or ``labels`` if given
    explicitly, e.g. to apply the training columns to a test set; unseen
    labels are ignored). Returns ``(X, labels)``.
    """
    profiles = list(profiles)
    if not profiles:
        raise ValueError("profiles is empty.")
    keys = {(pr.order, pr.criterion) for pr in profiles}
    if len(keys) > 1:
        raise ValueError(f"All profiles must share the same order and criterion: {keys}")
    if labels is None:
        labels = sorted(set().union(*(set(pr.labels.tolist()) for pr in profiles)))
    labels = list(labels)
    col = {lab: j for j, lab in enumerate(labels)}
    X = np.zeros((len(profiles), len(labels)), dtype=np.float64)
    for i, pr in enumerate(profiles):
        vals = pr.frequencies() if normalize else pr.counts
        for lab, v in zip(pr.labels.tolist(), vals):
            j = col.get(lab)
            if j is not None:
                X[i, j] = v
    return X, np.array(labels, dtype=object)
