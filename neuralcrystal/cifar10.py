"""CIFAR-10: a 32 × 32 colour photo in, one of ten classes out.

The photo is contrast-normalised (re-centred on mid-grey, its spread set to the file's `gcn` of full scale) and drawn as MEGA COLOUR
PIXELS: every photo pixel becomes a 2 × 2 of colour regions (red, green / blue, 255 − luma), each region an area-dither cell on the DMD
at the crystal's dither levels. Four passes through the crystal, then twenty squares in differential pairs; class k scores
(p⁺ − p⁻) / (p⁺ + p⁻). With flip=True the mirrored photo is run too and the two scores are added (the published "with flip" number).

    c = neuralcrystal.load("cifar10-crystal.safetensors")
    classes, scores = cifar10.classify(c, photo)     # photo: uint8 [32, 32, 3] or [B, 32, 32, 3]
    cifar10.LABELS[classes[0]]
"""
import numpy as np
import torch

from . import detectors

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
PLANE = 32


def options(crystal):
    m = crystal.m; o = dict(m.get("cifar") or (m.get("tr") or {}).get("cifar") or {})
    if (o.get("code") or "mega") != "mega": raise NotImplementedError(f"frame code {o.get('code')!r}: only 'mega' is supported")
    return {"frame": int(o.get("frame") or 256), "gcn": float(o.get("gcn") or 0), "p4": str(o.get("p4") or "dark"), "levels": crystal.q_levels}


def ranks(R):                                                                 # the R × R Bayer order: which dither cell lights first
    M = np.zeros((1, 1), dtype=np.int64); s = 1
    while s < R:
        N = np.zeros((2 * s, 2 * s), dtype=np.int64)
        for y in range(2 * s):
            for x in range(2 * s): N[y, x] = 4 * M[y % s, x % s] + [0, 2, 3, 1][(2 if y >= s else 0) + (1 if x >= s else 0)]
        M = N; s *= 2
    return torch.tensor(M)


def gcn(x, std):                                                              # per photo: mean to mid-grey, spread (all 3,072 values, n − 1) to std of full scale
    if not std: return x
    mu = x.mean(dim=(1, 2, 3), keepdim=True); sd = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
    return torch.clamp(127.5 + 255.0 * std * (x - mu) / sd, 0.0, 255.0)


def planes(x, p4="dark"):                                                     # [B,32,32,3] 0..255 → [B,4,32,32]: R, G, B and the fourth plane
    R, G, Bl = x[..., 0], x[..., 1], x[..., 2]; Y = 0.299 * R + 0.587 * G + 0.114 * Bl
    P4 = Y if p4 == "luma" else 255.0 - Y if p4 == "dark" else torch.full_like(Y, 255.0) if p4 == "carrier" else torch.zeros_like(Y)
    return torch.stack([R, G, Bl, P4], 1)


def mega(P, levels, frame):                                                   # [B,4,32,32] in 0..1 → [B,frame,frame], each region an area-dither cell
    R = frame // 64; B = P.shape[0]; Lc = levels - 1; Q = R * R * Lc; q = torch.round(P.clamp(0, 1) * Q).to(torch.int64)
    base = q // (R * R); extra = q % (R * R); rk = ranks(R)
    cells = base[..., None, None] + (rk[None, None, None, None] < extra[..., None, None]).to(torch.int64)
    v = cells.to(torch.float32) / Lc
    return v.reshape(B, 2, 2, PLANE, PLANE, R, R).permute(0, 3, 1, 5, 4, 2, 6).reshape(B, frame, frame).contiguous()


def frame(crystal, photo):
    """uint8 [32, 32, 3] / [B, 32, 32, 3] → float32 [B, frame, frame] in 0..1, the DMD picture this crystal was trained on."""
    o = options(crystal); x = torch.as_tensor(np.array(photo)).to(torch.float32)
    if x.dim() == 3: x = x[None]
    if tuple(x.shape[1:]) != (32, 32, 3): raise ValueError(f"a CIFAR-10 photo is 32 × 32 × 3, got {tuple(x.shape[1:])}")
    return mega(planes(gcn(x, o["gcn"]), o["p4"]) / 255.0, o["levels"], o["frame"])


def classify(crystal, photo, flip=False):
    """→ (classes [B] long, scores [B, 10]); flip=True adds the mirrored photo's scores."""
    E, _ = crystal.run(frame(crystal, photo)); s = detectors.scores(crystal, E)
    if flip:
        x = torch.as_tensor(np.array(photo)); x = x[None] if x.dim() == 3 else x
        E2, _ = crystal.run(frame(crystal, x.flip(2))); s = s + detectors.scores(crystal, E2)
    return s.argmax(1), s
