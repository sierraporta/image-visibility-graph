"""Tests for ivg against independent references.

Run with ``pytest tests/`` or, without pytest, ``python tests/test_ivg.py``.
"""

from __future__ import annotations

import itertools
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ivg  # noqa: E402
from ivg.patches import patch_layout  # noqa: E402

RNG = np.random.default_rng(12345)


# ---------------------------------------------------------------- references
def visible_bruteforce(x, a, b, criterion):
    """Literal definition from the paper, with exact arithmetic (Fraction)."""
    if a > b:
        a, b = b, a
    xa, xb = Fraction(x[a]), Fraction(x[b])
    for k in range(a + 1, b):
        xk = Fraction(x[k])
        if criterion == "horizontal":
            if not xk < min(xa, xb):
                return False
        elif not xk < xa + (xb - xa) * (k - a) / (b - a):
            return False
    return True


def graph_bruteforce(img, criterion):
    """Brute-force IVG/IHVG over all pixel pairs."""
    H, W = img.shape
    edges = set()
    for (i, j), (k, l) in itertools.combinations(itertools.product(range(H), range(W)), 2):
        di, dj = k - i, l - j
        if not (di == 0 or dj == 0 or abs(di) == abs(dj)):
            continue
        n = max(abs(di), abs(dj))
        si, sj = np.sign(di), np.sign(dj)
        line = [img[i + t * si, j + t * sj] for t in range(n + 1)]
        if visible_bruteforce(line, 0, n, criterion):
            u, v = i * W + j, k * W + l
            edges.add((min(u, v), max(u, v)))
    return edges


def paper_algorithm1(I, criterion="horizontal"):
    """Literal transcription of the paper's Algorithm 1 (PatchProfile), p = 3."""
    if criterion == "horizontal":
        M1 = lambda x0, x1, x2: (x1 < x0) and (x2 > x1)  # noqa: E731
    else:
        M1 = lambda x0, x1, x2: x2 > 2 * x1 - x0  # noqa: E731  (Table I, VG)
    I = np.asarray(I, dtype=np.float64)
    Z = np.zeros(256, dtype=np.int64)
    for i in range(I.shape[0] - 2):
        for j in range(I.shape[1] - 2):
            V = [
                M1(I[i, j], I[i, j + 1], I[i, j + 2]),
                M1(I[i + 1, j], I[i + 1, j + 1], I[i + 1, j + 2]),
                M1(I[i + 2, j], I[i + 2, j + 1], I[i + 2, j + 2]),
                M1(I[i, j], I[i + 1, j], I[i + 2, j]),
                M1(I[i, j + 1], I[i + 1, j + 1], I[i + 2, j + 1]),
                M1(I[i, j + 2], I[i + 1, j + 2], I[i + 2, j + 2]),
                M1(I[i, j], I[i + 1, j + 1], I[i + 2, j + 2]),
                M1(I[i, j + 2], I[i + 1, j + 1], I[i + 2, j]),
            ]
            L = 1 + sum(int(v) * 2 ** (8 - t) for t, v in enumerate(V, start=1))
            Z[L - 1] += 1
    return Z


# ---------------------------------------------------------------- motifs
def test_order3_motifs_match_table_I():
    for x in itertools.product(range(5), repeat=3):
        x0, x1, x2 = x
        assert ivg.motif_codes(np.array(x), "horizontal") == int(x1 < x0 and x2 > x1)
        assert ivg.motif_codes(np.array(x), "natural") == int(x2 > 2 * x1 - x0)


def test_motif_codes_match_bruteforce_edges():
    for crit in ivg.CRITERIA:
        for L in (3, 4, 5, 6):
            X = RNG.integers(0, 4, size=(300, L))  # with ties
            codes = ivg.motif_codes(X, crit)
            for x, c in zip(X, codes):
                ref = sorted(
                    (a, b) for a in range(L) for b in range(a + 1, L) if visible_bruteforce(x, a, b, crit)
                )
                assert ivg.code_to_edges(int(c), L) == ref


def test_hvg_admissible_motifs_are_catalan():
    assert [len(ivg.admissible_motifs(L, "horizontal")) for L in (2, 3, 4, 5, 6)] == [1, 2, 5, 14, 42]


# ---------------------------------------------------------------- 1D and 2D graphs
def test_1d_graph_matches_bruteforce_and_networkx():
    import networkx as nx

    for crit in ivg.CRITERIA:
        for _ in range(30):
            x = RNG.integers(0, 6, size=RNG.integers(2, 25))
            e = {tuple(r) for r in ivg.visibility_graph_edges(x, crit).tolist()}
            ref = {
                (a, b) for a in range(len(x)) for b in range(a + 1, len(x)) if visible_bruteforce(x, a, b, crit)
            }
            assert e == ref
    x = RNG.random(60)
    G = nx.visibility_graph(x)
    assert {tuple(sorted(e)) for e in G.edges()} == {
        tuple(r) for r in ivg.visibility_graph_edges(x, "natural").tolist()
    }


def test_image_graph_matches_bruteforce():
    for crit in ivg.CRITERIA:
        for shape in [(1, 1), (1, 7), (6, 1), (5, 5), (4, 7), (7, 3)]:
            img = RNG.integers(0, 4, size=shape)  # with ties
            G = ivg.image_visibility_graph(img, crit)
            got = {tuple(r) for r in G.edges.tolist()}
            assert len(got) == G.n_edges  # no duplicate edges
            assert got == graph_bruteforce(img, crit), (crit, shape)


def test_hvg_white_noise_link_probability():
    # For continuous i.i.d. noise, P(edge at distance d) = 2/[d(d+1)] in an HVG.
    x = RNG.random(200_000)
    e = ivg.visibility_graph_edges(x, "horizontal")
    dist = np.bincount(e[:, 1] - e[:, 0])
    for d in (1, 2, 3, 4):
        p_hat = dist[d] / (len(x) - d)
        assert abs(p_hat - 2 / (d * (d + 1))) < 0.01


# ---------------------------------------------------------------- patches
def test_patch_profile_order3_matches_paper_algorithm():
    for crit in ivg.CRITERIA:
        for shape in [(3, 3), (5, 7), (20, 31)]:
            img = RNG.integers(0, 256, size=shape, dtype=np.uint8)
            prof = ivg.patch_profile(img, 3, crit)
            assert np.array_equal(prof.to_dense(), paper_algorithm1(img, crit))
            assert prof.n_patches == (shape[0] - 2) * (shape[1] - 2)


def test_paper_figure4_label():
    d = ivg.decode_label(135, 3)
    assert [d[k] for k in ["r1", "r2", "r3", "c1", "c2", "c3", "d1", "d2"]] == [1, 0, 0, 0, 0, 1, 1, 0]
    assert ivg.label_bits(3) == 8 and ivg.n_labels(3) == 256


def test_patch_layout_counts():
    for p in (3, 4, 5, 6):
        lay = patch_layout(p)
        assert len(lay) == 2 * p + 2 + 4 * (p - 2)
        assert sum(ln.length == p for ln in lay) == 2 * p + 2
        for o in range(1, p - 1):
            assert sum(ln.length == p - o for ln in lay) == 4
        # Every line is straight within the patch and there are no repeats
        seen = set()
        for ln in lay:
            step = np.diff(ln.coords, axis=0)
            assert len({tuple(s) for s in step.tolist()}) <= 1
            key = frozenset(map(tuple, ln.coords.tolist()))
            assert key not in seen
            seen.add(key)


def test_patch_equals_induced_subgraph_of_full_graph():
    """A VP_p patch is the induced subgraph of the IVG: cross-check between the two modules."""
    for crit in ivg.CRITERIA:
        for p in (3, 4, 5):
            img = RNG.integers(0, 5, size=(p + 3, p + 4))
            H, W = img.shape
            edges = {tuple(r) for r in ivg.image_visibility_graph(img, crit).edges.tolist()}
            codes = ivg.patch_codes(img, p, crit)
            for i in range(H - p + 1):
                for j in range(W - p + 1):
                    for li, ln in enumerate(patch_layout(p)):
                        pix = [(i + r) * W + (j + c) for r, c in ln.coords.tolist()]
                        ref = sorted(
                            (a, b)
                            for a in range(ln.length)
                            for b in range(a + 1, ln.length)
                            if (min(pix[a], pix[b]), max(pix[a], pix[b])) in edges
                        )
                        assert ivg.code_to_edges(int(codes[i, j, li]), ln.length) == ref


def test_label_roundtrip_and_label_map():
    img = RNG.integers(0, 256, size=(15, 18))
    for p in (3, 4):
        prof = ivg.patch_profile(img, p, "natural")
        lm = ivg.patch_label_map(img, p, "natural")
        assert np.array_equal(prof.label_map(), lm)
        codes = ivg.patch_codes(img, p, "natural")
        for lab, row in zip(prof.labels, prof.codes):
            assert list(ivg.decode_label(lab, p).values()) == row.tolist()
        assert lm.min() >= 1 and lm.max() <= ivg.n_labels(p)
        assert codes.shape[:2] == lm.shape
    prof5 = ivg.patch_profile(img, 5, "horizontal")  # 88 bits: labels as Python ints
    assert ivg.label_bits(5) == 88
    assert prof5.n_patches == 11 * 14
    lab = prof5.labels[0]
    assert list(ivg.decode_label(lab, 5).values()) == prof5.codes[0].tolist()


def test_float_images_are_not_truncated():
    img = RNG.random((40, 40))
    prof_f = ivg.patch_profile(img, 3, "horizontal")
    prof_i = ivg.patch_profile(np.round(img * 1e6).astype(np.int64), 3, "horizontal")
    assert len(prof_f) > 100
    assert np.array_equal(prof_f.to_dense(), prof_i.to_dense())


def test_hvp3_admissible_patches_is_188():
    # The paper reports 188 distinct HVP_3; continuous noise produces all of them.
    prof = ivg.patch_profile(RNG.random((300, 300)), 3, "horizontal")
    assert len(prof) == 188


def test_horizontal_invariant_to_monotone_transform():
    img = RNG.integers(1, 200, size=(30, 30)).astype(float)
    a = ivg.patch_profile(img, 4, "horizontal")
    b = ivg.patch_profile(np.log(img) * 3 + 7, 4, "horizontal")
    assert a.labels.tolist() == b.labels.tolist() and np.array_equal(a.counts, b.counts)
    ga = ivg.image_visibility_graph(img, "horizontal").edges
    gb = ivg.image_visibility_graph(img ** 3, "horizontal").edges
    assert np.array_equal(ga, gb)


def test_profile_matrix_alignment():
    imgs = [RNG.integers(0, 256, size=(12, 12)) for _ in range(3)]
    profs = [ivg.patch_profile(im, 4, "horizontal") for im in imgs]
    X, labels = ivg.profile_matrix(profs)
    assert X.shape == (3, len(labels))
    assert np.allclose(X.sum(1), 1.0)
    X2, _ = ivg.profile_matrix(profs[:1], labels=labels)
    assert np.allclose(X2[0], X[0])


def test_degree_distribution_sums_to_one():
    G = ivg.image_visibility_graph(RNG.random((20, 25)), "natural")
    assert np.isclose(G.degree_distribution().sum(), 1.0)
    assert G.degree_map().shape == (20, 25)
    assert G.degrees().sum() == 2 * G.n_edges
    A = G.adjacency()
    assert (A != A.T).nnz == 0 and A.nnz == 2 * G.n_edges


def test_input_validation():
    def raises(exc, f, *a, **k):
        try:
            f(*a, **k)
        except exc:
            return
        raise AssertionError(f"{f.__name__} did not raise {exc.__name__}")

    raises(ValueError, ivg.patch_profile, np.zeros((5, 5, 3)), 3)
    raises(ValueError, ivg.patch_profile, np.zeros((5, 5)), 2)
    raises(ValueError, ivg.patch_profile, np.zeros((2, 5)), 3)
    raises(ValueError, ivg.patch_profile, np.zeros((5, 5)), 3, "diagonal")
    raises(ValueError, ivg.patch_profile, np.full((5, 5), np.nan), 3)
    raises(FileNotFoundError, ivg.load_gray, ROOT / "does_not_exist.png")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, f in tests:
        try:
            f()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
