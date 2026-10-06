"""Pictures of what the light does: the DMD frame going in, the camera's pictures between passes, and the light on the exit face.

Every function returns a uint8 RGB array [H, W, 3]; save(arr, path) writes a PNG (needs Pillow). Brightness is drawn in the laser's
red-orange (617 nm), as the simulator at neuralcrystal.com shows it; pass color=None for grey. Only the drawing is colored: what goes
into the crystal is always plain brightness, 0..1.
"""
import numpy as np
import torch

from . import detectors


def _u8(a): return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


LASER = (255, 78, 36)                                                       # 617 nm, as drawn on neuralcrystal.com
MARK, CHOSEN = (40, 178, 146), (240, 244, 246)                              # outlines: the readout squares / legal moves, and the answer


def _rgb(g, color=LASER):
    g = np.clip(g, 0, 1)
    if color is None: color = (255, 255, 255)
    return (g[..., None] * np.array(color, dtype=np.float64) + 0.5).astype(np.uint8)


def frame(f, scale=4, color=LASER):
    """A DMD frame or a hidden picture [F, F] in 0..1, each pixel scale × scale."""
    a = f.detach().cpu().float().numpy() if torch.is_tensor(f) else np.asarray(f, dtype=np.float32)
    return np.kron(_rgb(a, color), np.ones((scale, scale, 1), dtype=np.uint8))


def exit_light(crystal, E, i=0, gamma=0.5, boxes=True, highlight=None, size=512, as_camera=True, color=LASER):
    """|E|² of sample i over the crystal's window, brightness ^gamma so faint light shows, resized to size × size.
    boxes: outline the readout squares (classifiers); highlight: a square index drawn in colour (the answer).
    as_camera: when the crystal's output relay turns the picture 180° on its way to the camera (MNIST's does), show it as the
    camera sees it, upright, rather than as it leaves the glass."""
    I = (E[i].real ** 2 + E[i].imag ** 2).detach().cpu().double().numpy()
    N = crystal.N; w = max(1, int(round(N * crystal.win_frac))); x0 = (N - w) // 2
    win = I[x0:x0 + w, x0:x0 + w]; win = (win / max(win.max(), 1e-30)) ** gamma
    img = _rgb(win, color)
    if boxes:
        try: R = detectors.rects(crystal)
        except ValueError: R = []
        for d, (bx, by, bw, bh) in enumerate(R):
            col = CHOSEN if d == highlight else MARK
            x, y = bx - x0, by - x0                                           # the window is centred, so rows and columns share the offset
            img[max(0, y):y + bh, max(0, x):max(0, x) + 1] = col; img[max(0, y):y + bh, x + bw - 1:x + bw] = col
            img[max(0, y):max(0, y) + 1, max(0, x):x + bw] = col; img[y + bh - 1:y + bh, max(0, x):x + bw] = col
    if as_camera and crystal.sensor_turned: img = img[::-1, ::-1].copy()
    if size and size != w:
        idx = (np.arange(size) * w / size).astype(int); img = img[idx][:, idx]
    return img


def light(I, gamma=0.5, scale=1, color=LASER):
    """A light picture [H, W] (any intensity: a traced surface, a move map) → RGB, scaled to its own brightest, ^gamma so faint light shows."""
    a = I.detach().cpu().double().numpy() if torch.is_tensor(I) else np.asarray(I, dtype=np.float64)
    a = (a / max(a.max(), 1e-30)) ** gamma
    return np.kron(_rgb(a, color), np.ones((scale, scale, 1), dtype=np.uint8)) if scale > 1 else _rgb(a, color)


def grid(rows, gap=6, row_labels=None, col_labels=None, font_px=22):
    """Rows of pictures (lists) stacked into one image, each row a strip. row_labels go down the left, col_labels across the top
    (centred over the first row's pictures); labels need Pillow."""
    strips = [strip(r, gap) for r in rows]; W = max(s.shape[1] for s in strips); H = sum(s.shape[0] for s in strips) + gap * (len(strips) - 1)
    out = np.full((H, W, 3), 24, np.uint8); y = 0
    for s in strips: out[y:y + s.shape[0], :s.shape[1]] = s; y += s.shape[0] + gap
    if not (row_labels or col_labels): return out
    from PIL import Image, ImageDraw, ImageFont
    try: font = ImageFont.load_default(size=font_px)
    except TypeError: font = ImageFont.load_default()
    left = (max(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(t, font=font) for t in row_labels) + 16) if row_labels else 0
    top = font_px + 14 if col_labels else 0
    img = Image.new("RGB", (W + int(left), H + top), (24, 24, 24)); img.paste(Image.fromarray(out), (int(left), top)); d = ImageDraw.Draw(img)
    if col_labels:
        x = int(left)
        for p, t in zip(rows[0], col_labels):
            d.text((x + p.shape[1] / 2, top / 2), t, fill=(220, 220, 220), font=font, anchor="mm"); x += p.shape[1] + gap
    if row_labels:
        y = top
        for r, t in zip(rows, row_labels):
            h = max(p.shape[0] for p in r); d.text((8, y + h / 2), t, fill=(220, 220, 220), font=font, anchor="lm"); y += h + gap
    return np.asarray(img)


def strip(pictures, gap=8):
    """Pictures side by side on a dark ground, tops aligned."""
    h = max(p.shape[0] for p in pictures); W = sum(p.shape[1] for p in pictures) + gap * (len(pictures) - 1)
    out = np.full((h, W, 3), 24, np.uint8); x = 0
    for p in pictures: out[:p.shape[0], x:x + p.shape[1]] = p; x += p.shape[1] + gap
    return out


def save(img, path):
    from PIL import Image
    Image.fromarray(img).save(path)
