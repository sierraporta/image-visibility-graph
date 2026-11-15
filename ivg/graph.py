"""Image Visibility Graph (IVG) and Image Horizontal Visibility Graph (IHVG).

Reference: Iacovacci, J. & Lacasa, L. (2019). Visibility graphs for image
processing. IEEE TPAMI 42(4), 974-987. Section II.A.

Each pixel is a node (flat index ``i*W + j``). Two pixels are linked if they
lie on the same row, column, diagonal or anti-diagonal (FCC class) and
"see" each other in the 1D visibility graph of that line:

* natural (IVG):     x_k < x_a + (x_b - x_a)(k - a)/(b - a)  for all a < k < b
* horizontal (IHVG): x_k < min(x_a, x_b)                     for all a < k < b
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import as_image, check_criterion

DIRECTIONS = ("rows", "cols", "diag", "antidiag")


def _line_index_matrix(H: int, W: int, direction: str) -> np.ndarray:
    """(n_lines, max_length) matrix of flat pixel indices; -1 = padding."""
    idx = np.arange(H * W, dtype=np.int64).reshape(H, W)
    if direction == "rows":
        return idx
    if direction == "cols":
        return np.ascontiguousarray(idx.T)
    if direction in ("diag", "antidiag"):
        src = idx if direction == "diag" else idx[:, ::-1]
        n = min(H, W)
        out = np.full((H + W - 1, n), -1, dtype=np.int64)
        for r, off in enumerate(range(-(H - 1), W)):
            d = np.diagonal(src, off)
            out[r, : d.size] = d
        return out
    raise ValueError(f"Unknown direction {direction!r}; use one of {DIRECTIONS}.")


def _visibility_edges_lines(X: np.ndarray, criterion: str):
    """Visibility edges in a batch of lines.

    X : (n_lines, n) float64, with NaN padding at the end of each line.
    Returns (line, a, b) with a < b for every edge.

    Iterates over the distance d = b - a, vectorizing over all lines and
    positions a at once:

    * natural: (a, a+d) visible  <=>  slope(a, a+d) > max_{0<c<d} slope(a, a+c)
    * horizontal: (a, a+d) visible <=> max(x_{a+1..a+d-1}) < min(x_a, x_{a+d})
    """
    n_lines, n = X.shape
    out_l, out_a, out_d = [], [], []
    if n < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    with np.errstate(invalid="ignore"):
        # d = 1: consecutive neighbours, always visible (if both exist)
        valid = ~np.isnan(X[:, :-1]) & ~np.isnan(X[:, 1:])
        l, a = np.nonzero(valid)
        out_l.append(l), out_a.append(a), out_d.append(np.ones_like(a))
        if criterion == "natural":
            run = X[:, 1:] - X[:, :-1]  # max slope to intermediate points, for d = 2
        else:
            run = X[:, 1:-1].copy() if n > 2 else None  # max of intermediate points
        for d in range(2, n):
            xa = X[:, : n - d]
            xb = X[:, d:]
            if criterion == "natural":
                slope = (xb - xa) / d
                prev = run[:, : n - d]
                vis = slope > prev
                run = np.fmax(prev, slope)
            else:
                prev = run[:, : n - d]
                vis = prev < np.minimum(xa, xb)
                if d + 1 < n:
                    run = np.fmax(prev[:, : n - d - 1], X[:, d : n - 1])
                # Pruning: once the intermediate maximum is >= x_a everywhere,
                # no further edges are possible (horizontal criterion only).
                if not np.any(prev < xa):
                    if vis.any():
                        l, a = np.nonzero(vis)
                        out_l.append(l), out_a.append(a), out_d.append(np.full_like(a, d))
                    break
            if vis.any():
                l, a = np.nonzero(vis)
                out_l.append(l), out_a.append(a), out_d.append(np.full_like(a, d))
    L = np.concatenate(out_l)
    A = np.concatenate(out_a)
    D = np.concatenate(out_d)
    return L, A, A + D


def visibility_graph_edges(series, criterion: str = "natural") -> np.ndarray:
    """Edges (E, 2) of the visibility graph (VG or HVG) of a 1D series."""
    criterion = check_criterion(criterion)
    x = np.asarray(series, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("series must be 1D.")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or infinite values.")
    _, a, b = _visibility_edges_lines(x[None, :], criterion)
    e = np.stack([a, b], 1)
    return e[np.lexsort((e[:, 1], e[:, 0]))]


@dataclass
class ImageVisibilityGraph:
    """Result of :func:`image_visibility_graph`.

    Attributes
    ----------
    shape : (H, W) of the image.
    criterion : "natural" (IVG) or "horizontal" (IHVG).
    edges : array (E, 2) of flat indices (i*W + j), with edges[:, 0] < edges[:, 1].
    directions : directions included.
    """

    shape: tuple[int, int]
    criterion: str
    edges: np.ndarray
    directions: tuple[str, ...]

    @property
    def n_nodes(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def degrees(self) -> np.ndarray:
        """Degree of each node, as a flat vector of length H*W."""
        return np.bincount(self.edges.ravel(), minlength=self.n_nodes)

    def degree_map(self) -> np.ndarray:
        """Degree plot / k-filtered image: degree of each pixel, shape (H, W)."""
        return self.degrees().reshape(self.shape)

    def degree_distribution(self, kmax: int | None = None) -> np.ndarray:
        """P(k): fraction of nodes with degree k, for k = 0..kmax.

        Fixing ``kmax`` yields equal-length vectors across images (degrees
        above kmax are dropped).
        """
        k = self.degrees()
        kmax = int(k.max()) if kmax is None else int(kmax)
        counts = np.bincount(k[k <= kmax], minlength=kmax + 1)
        return counts / self.n_nodes

    def adjacency(self):
        """Sparse adjacency matrix (``scipy.sparse.csr_matrix``)."""
        from scipy.sparse import coo_matrix

        n = self.n_nodes
        e = self.edges
        data = np.ones(2 * len(e), dtype=np.int8)
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    def to_networkx(self, img=None):
        """``networkx`` graph; nodes labelled (i, j), with a ``value`` attribute if ``img`` is given."""
        import networkx as nx

        H, W = self.shape
        G = nx.Graph()
        vals = None if img is None else np.asarray(img).reshape(-1)
        for flat in range(H * W):
            attrs = {} if vals is None else {"value": vals[flat].item()}
            G.add_node(divmod(flat, W), **attrs)
        G.add_edges_from((divmod(int(u), W), divmod(int(v), W)) for u, v in self.edges)
        return G


def image_visibility_graph(
    img, criterion: str = "natural", directions=DIRECTIONS
) -> ImageVisibilityGraph:
    """Build the IVG (``criterion="natural"``) or IHVG (``"horizontal"``) of an image.

    Cost: O(N^3) vectorized operations for an N×N image in the worst case
    (the IHVG usually stops earlier thanks to pruning).
    """
    criterion = check_criterion(criterion)
    directions = tuple(directions)
    for d in directions:
        if d not in DIRECTIONS:
            raise ValueError(f"Unknown direction {d!r}; use one of {DIRECTIONS}.")
    I = as_image(img)
    H, W = I.shape
    flat = I.ravel()
    chunks = []
    for direction in directions:
        idx = _line_index_matrix(H, W, direction)
        X = np.where(idx >= 0, flat[np.maximum(idx, 0)], np.nan)
        l, a, b = _visibility_edges_lines(X, criterion)
        u, v = idx[l, a], idx[l, b]
        chunks.append(np.stack([np.minimum(u, v), np.maximum(u, v)], 1))
    edges = np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.int64)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    return ImageVisibilityGraph(shape=(H, W), criterion=criterion, edges=edges, directions=directions)


def degree_map(img, criterion: str = "natural") -> np.ndarray:
    """Shortcut: k-filtered image (degree of each pixel in the IVG/IHVG)."""
    return image_visibility_graph(img, criterion).degree_map()
