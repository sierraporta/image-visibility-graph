"""Compare ivg against the reference MATLAB code outputs (run_reference.m)."""
import glob
import os
import sys

import numpy as np

try:
    import ivg
except ImportError:  # running from a source checkout without installing
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    import ivg

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def edge_set(E):
    E = np.atleast_2d(E).astype(np.int64) - 1          # MATLAB nodes are 1-based, row-major
    return {(min(a, b), max(a, b)) for a, b in E.tolist()}, len(E)

def lattice(H, W):
    s = set()
    for i in range(H):
        for j in range(W):
            u = i * W + j
            for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
                k, l = i + di, j + dj
                if 0 <= k < H and 0 <= l < W:
                    s.add((u, k * W + l))
    return s

ok = True
for path in sorted(glob.glob("in/*.csv")):
    name = os.path.basename(path)[:-4]
    I = np.loadtxt(path, delimiter=",")
    line = [f"{name:9s}"]
    for crit, tag in (("horizontal", "Zh"), ("natural", "Zn")):
        Zm = np.loadtxt(f"out/{name}_{tag}.csv", delimiter=",")
        Zp = ivg.patch_profile(I, 3, crit).to_dense(normalize=True, numbering="matlab")
        same = np.allclose(Zm, Zp, rtol=0, atol=1e-12)
        ok &= same
        line.append(f"patches {crit[:5]}: {'OK' if same else 'DIFF'}")
    if os.path.exists(f"out/{name}_Eh_T.csv"):
        H, W = I.shape
        lat = lattice(H, W)
        for crit, c in (("horizontal", "h"), ("natural", "n")):
            ours = {tuple(e) for e in ivg.image_visibility_graph(I, crit).edges.tolist()}
            for flag in ("T", "F"):
                ref, n_rows = edge_set(np.loadtxt(f"out/{name}_E{c}_{flag}.csv", delimiter=","))
                dup = n_rows - len(ref)
                if ref == ours:
                    verdict = "= ours (with lattice)"
                elif ref == ours - lat:
                    verdict = "= ours without lattice"
                else:
                    ok = False
                    verdict = f"DIFF (+{len(ref - ours)} / -{len(ours - ref)})"
                line.append(f"graph {crit[:5]} lattice={flag}: {verdict}" + (f" [dup {dup}]" if dup else ""))
    print(" | ".join(line))
print("\nALL MATCH" if ok else "\nMISMATCHES FOUND")
sys.exit(0 if ok else 1)
