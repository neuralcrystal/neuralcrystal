"""The classifiers' readout: squares on the exit face, each collecting the light that lands on it.

A crystal file names its layout in `det` (in its problem block, and in `tr` for older readers). Centres and half-sides are in window
widths, rows downward. "padrot": ten squares in a phone-pad pattern turned 180° (MNIST: the brightest square is the digit).
"diffwin": twenty squares in differential pairs (CIFAR-10: class k scores (p⁺ − p⁻) / (p⁺ + p⁻), the largest wins).
"""
import math
import torch

PAD = [[0, 0.375], [-0.25, -0.375], [0, -0.375], [0.25, -0.375], [-0.25, -0.125], [0, -0.125], [0.25, -0.125], [-0.25, 0.125], [0, 0.125], [0.25, 0.125]]
LAYOUTS = {
    "padrot": ([[-x, -y] for x, y in PAD], 0.10),
    "diffwin": ([[x, y] for y in (-0.21, -0.07, 0.07, 0.21) for x in (-0.28, -0.14, 0, 0.14, 0.28)], 0.055),   # +0..4, +5..9, −0..4, −5..9
}


def rnd(x): return int(math.floor(x + 0.5))


def layout(crystal):
    m = crystal.m; p = m.get(m.get("problem") or "", {}) if isinstance(m.get(m.get("problem") or ""), dict) else {}
    det = p.get("det") or (m.get("tr") or {}).get("det")
    if det not in LAYOUTS: raise ValueError(f"detector layout {det!r} is not a classifier layout ({', '.join(LAYOUTS)})")
    return det


def rects(crystal):
    """[(x0, y0, w, h)] of each square in samples on the crystal's N × N grid."""
    centres, h = LAYOUTS[layout(crystal)]; n = crystal.N; w = n * crystal.win_frac; c = (n - 1) / 2
    return [(rnd(c + (cx - h) * w), rnd(c + (cy - h) * w), rnd(2 * h * w), rnd(2 * h * w)) for cx, cy in centres]


def masks(crystal):
    """[squares, N, N] 0/1 on the crystal's device."""
    R = rects(crystal); N = crystal.N; D = torch.zeros(len(R), N, N)
    for d, (x0, y0, w, h) in enumerate(R): D[d, max(0, y0):min(N, y0 + h), max(0, x0):min(N, x0 + w)] = 1
    return D.to(crystal.dev)


def light(crystal, E):
    """The light on each square [B, squares] from the exit field E [B, N, N]."""
    I = (E.real * E.real + E.imag * E.imag).reshape(E.shape[0], -1); D = masks(crystal)
    return I @ D.reshape(D.shape[0], -1).T + 1e-12


def scores(crystal, E):
    """Class scores [B, 10]: the squares' light (ten squares), or the pairs' (p⁺ − p⁻) / (p⁺ + p⁻) (twenty)."""
    p = light(crystal, E)
    if p.shape[1] == 20: pp, pn = p[:, :10], p[:, 10:]; return (pp - pn) / (pp + pn)
    return p
