# image-visibility-graph

A Python implementation of **Image Visibility Graphs** and **Visibility Patches**, the
image-analysis method introduced by Iacovacci & Lacasa,
*Visibility graphs for image processing*, IEEE TPAMI 42(4), 974–987 (2020)
([doi:10.1109/TPAMI.2019.2891742](https://doi.org/10.1109/TPAMI.2019.2891742),
[arXiv:1804.07125](https://arxiv.org/abs/1804.07125)).

The package is imported as `ivg`. Its only hard dependency is NumPy, and every computation is vectorized.

## Features

- **Both visibility criteria** everywhere, selected with `criterion=`:

  | `criterion` | Graph | Local features | Order-3 motif (Table I of the paper) |
  |---|---|---|---|
  | `"natural"` | IVG | VP<sub>p</sub> | x₂ > 2x₁ − x₀ |
  | `"horizontal"` | IHVG | HVP<sub>p</sub> | x₁ < min(x₀, x₂) |

- **Visibility Patch profile Z** (local features) of any order *p* ≥ 3, including the
  4(*p*−2) off-diagonal motifs that the paper adds for *p* ≥ 4.
- **Full image graph** (global features): pixels linked along rows, columns, diagonals
  and anti-diagonals. Gives the degree map ("k-filtered image"), the degree distribution
  P(k), and export to a SciPy sparse matrix or NetworkX graph.
- **Interpretable labels**: `decode_label` tells you which rows, columns and diagonals
  carry which motif.
- **Exact comparisons**: images are handled as `float64`, so integer and real-valued
  images (e.g. normalized to [0, 1]) are processed without truncation.

## Installation

```bash
pip install git+https://github.com/sierraporta/image-visibility-graph
```

or, for development:

```bash
git clone https://github.com/sierraporta/image-visibility-graph
cd image-visibility-graph
python -m venv .venv && source .venv/bin/activate
pip install -e ".[extras,test,examples]"
```

The optional extras are NetworkX, SciPy, pandas, OpenCV/Pillow (image loading) and Matplotlib/Jupyter for the examples.

## Quick start

```python
import ivg

img = ivg.load_gray("my_image.png")            # 2D grayscale array

# Local features: Horizontal Visibility Patches of order 3 (256 possible patches)
prof = ivg.patch_profile(img, order=3, criterion="horizontal")
Z = prof.to_dense(normalize=True)              # feature vector of length 256
prof.most_common(5)                            # [(label, count), ...]
ivg.decode_label(135, order=3)                 # {'r1': 1, 'r2': 0, ..., 'd1': 1, 'd2': 0}

# Global features: the full Image Visibility Graph
G = ivg.image_visibility_graph(img, criterion="natural")
K = G.degree_map()                             # k-filtered image, same shape as img
Pk = G.degree_distribution(kmax=100)           # fixed-length P(k)
A = G.adjacency()                              # scipy.sparse CSR matrix
```

For a feature matrix over several images (rows aligned to a common label set):

```python
profiles = [ivg.patch_profile(im, order=3, criterion="horizontal") for im in images]
X, labels = ivg.profile_matrix(profiles)       # X.shape == (n_images, n_labels)
```

A complete example is in [`examples/example_ivg.ipynb`](examples/example_ivg.ipynb). It runs on synthetic
textures with tunable spatial correlation and on a real solar image from SDO/HMI, comparing an active
region with the quiet Sun.

## Method

### Image Visibility Graph

Every pixel is a node. Two pixels are linked if they lie on the same row, column,
diagonal or anti-diagonal and "see" each other in the 1D visibility graph of that line:

- natural: x<sub>k</sub> < x<sub>a</sub> + (x<sub>b</sub> − x<sub>a</sub>)(k − a)/(b − a) for all a < k < b
- horizontal: x<sub>k</sub> < min(x<sub>a</sub>, x<sub>b</sub>) for all a < k < b

### Visibility Patches and label convention

A Visibility Patch of order *p* is the subgraph of the IVG/IHVG induced by a *p*×*p*
window, slid over the image with stride 1. The patch is fully determined by the
*sequential visibility motif* of each line of the window, in this order:

```
V = [r1..rp, c1..cp, d1, d2, d1+1, d1-1, d2+1, d2-1, d1+2, d1-2, ...]
```

Here `d1` is the main diagonal and `d2` the anti-diagonal, which runs from top-right to
bottom-left as in the paper. `d1±o` and `d2±o` are the off-diagonals shifted by *o*.
Each line's motif is encoded by which of its non-adjacent pairs (a, b), b − a ≥ 2, are
visible. The patch label is `1 + concatenated bits`, with the first line as the most
significant bits.

For *p* = 3 this is **exactly the paper's encoding**:
L = 1 + Σ V<sub>i</sub> 2<sup>8−i</sup>, with 256 possible labels.
For example, V = [1,0,0,0,0,1,1,0] gives L = 135 (Fig. 4 of the paper).

### A note on higher orders

For *p* ≥ 4, each line can take more than two motifs, and the off-diagonals add more
lines. The label space therefore grows very fast (2<sup>34</sup> labels for *p* = 4,
2<sup>88</sup> for *p* = 5), and real images use only a tiny fraction of it.
`PatchProfile` stores profiles sparsely, `profile_matrix` aligns them across images,
and `label_map()` / `decode_label()` work for any order. As a feature vector, *p* = 3 is
usually the practical choice, which is also the order used in the paper.

## API overview

| Function / class | Purpose |
|---|---|
| `patch_profile(img, order, criterion)` → `PatchProfile` | Visibility Patch profile (sparse; `.to_dense()`, `.frequencies()`, `.to_series()`, `.label_map()`, `.most_common()`) |
| `profile_matrix(profiles, normalize, labels)` | Aligned feature matrix for several images |
| `patch_label_map(img, order, criterion)` | Label of the patch anchored at every pixel (*p* ≤ 4) |
| `patch_codes(img, order, criterion)` | Raw per-line motif codes of every patch |
| `decode_label(label, order)` / `decode_label_edges(...)` | Interpret a label |
| `patch_layout(order)`, `label_bits(order)`, `n_labels(order)` | Patch geometry and label-space size |
| `image_visibility_graph(img, criterion, directions)` → `ImageVisibilityGraph` | Full IVG/IHVG (`.edges`, `.degrees()`, `.degree_map()`, `.degree_distribution()`, `.adjacency()`, `.to_networkx()`) |
| `degree_map(img, criterion)` | Shortcut for the k-filtered image |
| `visibility_graph_edges(series, criterion)` | VG/HVG of a 1D series |
| `motif_codes(lines, criterion)`, `code_to_edges`, `admissible_motifs` | Sequential visibility motifs |
| `load_gray(path)` | Load an image as grayscale (raises if the file is missing) |

## Validation

The test suite (`python tests/test_ivg.py`, or `pytest`) checks the implementation
against independent references:

- **Algorithm 1 of the paper:** a literal transcription for *p* = 3 gives identical
  profiles, for both criteria.
- **Fig. 4 of the paper:** the example decodes to label 135.
- **Full graph:** matches a brute-force construction over all pixel pairs in exact
  rational arithmetic, including images with ties.
- **1D graph:** the natural VG matches `networkx.visibility_graph`.
- **White noise:** HVG link probabilities follow the exact law 2/[d(d+1)] (Luque et al., 2009).
- **Patches of order 3, 4 and 5:** each patch equals the subgraph that the full graph
  induces on its window. This cross-checks the two modules against each other.
- **Admissible motifs and patches:**
  - The number of admissible HVG motifs follows the Catalan numbers (1, 2, 5, 14, 42).
  - Continuous noise produces exactly the 188 distinct HVP<sub>3</sub> patches reported in the paper.
- **Invariance:** the horizontal criterion is invariant under strictly increasing intensity transforms.

## Performance

Timings on a laptop CPU (NumPy only):

| Image | Patch profile *p* = 3 | Patch profile *p* = 5 | Full graph |
|---|---|---|---|
| 512 × 512 | < 0.1 s | ~1.5 s | 1.6 s (IHVG) / 2.7 s (IVG) |
| 1024 × 1024 | ~0.2 s | ~4.5 s | 12 s (IHVG) / 15 s (IVG) |

Patch profiles scale linearly with the number of pixels. The full graph costs O(N³) for
an N × N image in the worst case.

## Related software

- [Jaia89/Image_Visibility](https://github.com/Jaia89/Image_Visibility): MATLAB
  functions by the paper's first author to extract IVGs and Visibility Patch profiles.
- [ts2vg](https://github.com/CarlosBergillos/ts2vg): fast natural and horizontal
  visibility graphs for **time series** (1D).
- [Lucas Lacasa's visibility graph page](https://sites.google.com/view/lucaslacasa/research-topics/visibility-graphs):
  original codes, e.g. MATLAB motif scripts.

## Citing

If you use this package, please cite the original method:

```bibtex
@article{iacovacci2020visibility,
  title   = {Visibility Graphs for Image Processing},
  author  = {Iacovacci, Jacopo and Lacasa, Lucas},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {42},
  number  = {4},
  pages   = {974--987},
  year    = {2020},
  doi     = {10.1109/TPAMI.2019.2891742}
}
```

Related foundational work:

- L. Lacasa, B. Luque, F. Ballesteros, J. Luque, J. C. Nuño, *From time series to complex
  networks: The visibility graph*, PNAS 105, 4972 (2008).
  [doi:10.1073/pnas.0709247105](https://doi.org/10.1073/pnas.0709247105)
- B. Luque, L. Lacasa, F. Ballesteros, J. Luque, *Horizontal visibility graphs: Exact
  results for random time series*, Phys. Rev. E 80, 046103 (2009).
  [doi:10.1103/PhysRevE.80.046103](https://doi.org/10.1103/PhysRevE.80.046103)
- J. Iacovacci, L. Lacasa, *Sequential visibility-graph motifs*, Phys. Rev. E 93,
  042309 (2016). [doi:10.1103/PhysRevE.93.042309](https://doi.org/10.1103/PhysRevE.93.042309)
- J. Iacovacci, L. Lacasa, *Sequential motif profile of natural visibility graphs*,
  Phys. Rev. E 94, 052309 (2016). [doi:10.1103/PhysRevE.94.052309](https://doi.org/10.1103/PhysRevE.94.052309)

## License

Code: [MIT](LICENSE) © 2026 David Sierra Porta.

Example data: the solar image in `examples/data/` is courtesy of NASA/SDO and the AIA, EVE,
and HMI science teams. SDO images are not copyrighted unless explicitly noted
([SDO data rules](https://sdo.gsfc.nasa.gov/data/rules.php)). See
[`examples/data/README.md`](examples/data/README.md).
