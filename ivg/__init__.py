"""ivg — Image Visibility Graphs (Iacovacci & Lacasa, IEEE TPAMI 2019).

Two visibility criteria, selectable in every function through
``criterion``:

* ``"natural"``    -> IVG  / Visibility Patches VP_p
* ``"horizontal"`` -> IHVG / Horizontal Visibility Patches HVP_p

Quick start
-----------
>>> import ivg
>>> img = ivg.load_gray("Lenna_(test_image).png")
>>> prof = ivg.patch_profile(img, order=3, criterion="horizontal")
>>> Z = prof.to_dense(normalize=True)            # profile Z (256 frequencies)
>>> G = ivg.image_visibility_graph(img, "natural")
>>> K = G.degree_map()                           # k-filtered image
>>> Pk = G.degree_distribution(kmax=100)         # global feature P(k)
"""

from ._common import CRITERIA, load_gray
from .graph import (
    DIRECTIONS,
    ImageVisibilityGraph,
    degree_map,
    image_visibility_graph,
    visibility_graph_edges,
)
from .motifs import admissible_motifs, code_to_edges, motif_codes, motif_pairs, n_pairs
from .patches import (
    PatchLine,
    PatchProfile,
    decode_label,
    decode_label_edges,
    label_bits,
    matlab_to_paper_label,
    n_labels,
    paper_to_matlab_label,
    patch_codes,
    patch_label_map,
    patch_layout,
    patch_profile,
    profile_matrix,
)

__version__ = "0.1.0"

__all__ = [
    "CRITERIA",
    "DIRECTIONS",
    "ImageVisibilityGraph",
    "PatchLine",
    "PatchProfile",
    "admissible_motifs",
    "code_to_edges",
    "decode_label",
    "decode_label_edges",
    "degree_map",
    "image_visibility_graph",
    "label_bits",
    "load_gray",
    "matlab_to_paper_label",
    "motif_codes",
    "motif_pairs",
    "n_labels",
    "n_pairs",
    "patch_codes",
    "patch_label_map",
    "patch_layout",
    "paper_to_matlab_label",
    "patch_profile",
    "profile_matrix",
    "visibility_graph_edges",
]
