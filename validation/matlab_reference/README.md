# Cross-validation against the reference MATLAB implementation

These scripts compare `ivg` with the MATLAB code written by the paper's first author,
Jacopo Iacovacci ([Jaia89/Image_Visibility](https://github.com/Jaia89/Image_Visibility)).
That code is licensed under GPL-3.0, so it is **not** included here: download it separately.

## How to run

You need MATLAB or GNU Octave (`brew install octave`, `apt install octave`, ...).

```bash
git clone https://github.com/Jaia89/Image_Visibility /path/to/Image_Visibility
cd validation/matlab_reference
python make_inputs.py                                   # writes in/*.csv
REFDIR=/path/to/Image_Visibility octave --no-gui --quiet run_reference.m   # writes out/*.csv
python compare.py
```

`run_reference.m` also runs the reference `visibilityPatches.m` on the `lena512.mat` image
that ships with the reference repository (patches only).

## Result (ivg 0.1.0, reference commit `bd7f493`, GNU Octave 11.3)

| Image | Patches HVP₃ | Patches VP₃ | IHVG edges | IVG edges |
|---|---|---|---|---|
| `ties12`, `ties30` (many repeated values) | identical | identical | identical | identical |
| `float40` (real-valued) | identical | identical | identical | identical |
| `noise64` (uint8 white noise) | identical | identical | identical | identical |
| `corr64`, `corr96` (correlated surfaces) | identical | identical | identical | identical |
| `lena512` (from the reference repo) | identical | identical | — | — |

Two differences in conventions, neither of which changes the results:

1. **Patch label numbering.** `visibilityPatches.m` builds the 8-bit string in the line order
   `[r1, c3, r3, c1, c2, r2, d1, d2]`, with the first element as the least significant bit.
   Algorithm 1 of the paper (and `ivg`) uses `[r1, r2, r3, c1, c2, c3, d1, d2]`, with the
   first element as the most significant bit. The 256 patches are the same; only their
   numbering differs. Use `profile.to_dense(numbering="matlab")` or
   `ivg.paper_to_matlab_label` to convert.
2. **The `lattice` flag of `imageVisibilityGraph.m`.** In the `'natural'` branch the flag
   behaves inverted: `lattice=true` omits the 8-neighbour edges and `lattice=false` includes
   them. The `'horizontal'` branch behaves as documented. `ivg` always includes them by
   default; use `image_visibility_graph(..., lattice=False)` to drop them.
