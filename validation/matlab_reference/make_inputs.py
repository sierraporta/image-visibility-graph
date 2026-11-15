"""Generate the test images used to compare ivg with the reference MATLAB code.

Writes CSV files to ``in/``: images with many ties, real-valued images,
white noise and spatially correlated surfaces.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(2026)


def correlated(n, beta):
    fx = np.fft.fftfreq(n)[:, None]
    fy = np.fft.fftfreq(n)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = 1
    z = np.real(np.fft.ifft2((rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / f ** (beta / 2)))
    return np.round(255 * (z - z.min()) / (z.max() - z.min()))


cases = {
    "ties12": rng.integers(0, 4, (12, 12)).astype(float),
    "ties30": rng.integers(0, 8, (30, 30)).astype(float),
    "float40": rng.random((40, 40)),
    "noise64": rng.integers(0, 256, (64, 64)).astype(float),
    "corr64": correlated(64, 2.5),
    "corr96": correlated(96, 1.5),
}
(HERE / "in").mkdir(exist_ok=True)
(HERE / "out").mkdir(exist_ok=True)
for name, img in cases.items():
    np.savetxt(HERE / "in" / f"{name}.csv", img, delimiter=",", fmt="%.17g")
print("wrote", ", ".join(cases))
