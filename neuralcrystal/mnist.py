"""MNIST: a handwritten digit in, the digit out.

The digit (28 × 28, white ink on black, as MNIST stores it) is drawn on a 32 × 32 frame two pixels in from the top-left, each pixel's
brightness the DMD's amplitude. One pass through the crystal; ten squares on the exit face, the brightest is the answer.

    c = neuralcrystal.load("mnist-crystal.safetensors")
    digit, scores = mnist.classify(c, img)          # img: uint8 [28, 28] or [B, 28, 28]
"""
import numpy as np
import torch

from . import detectors

FRAME, PAD = 32, 2


def frame(img):
    """uint8 or float 0..255 [28, 28] / [B, 28, 28] → float32 [B, 32, 32] in 0..1."""
    x = torch.as_tensor(np.array(img)).to(torch.float32)
    if x.dim() == 2: x = x[None]
    if x.shape[-2:] != (28, 28): raise ValueError(f"an MNIST digit is 28 × 28, got {tuple(x.shape[-2:])}")
    out = torch.zeros(x.shape[0], FRAME, FRAME); out[:, PAD:PAD + 28, PAD:PAD + 28] = x / 255.0
    return out


def classify(crystal, img):
    """→ (digits [B] long, scores [B, 10]: the light on each digit's square)."""
    E, _ = crystal.run(frame(img))
    s = detectors.scores(crystal, E)
    return s.argmax(1), s
