"""A crystal's input and output, drawn large and labeled (MNIST, CIFAR-10 and checkers; chess has its own: chess_input_output.py).

    python examples/input_output.py mnist-crystal.safetensors examples/assets/digit-4.png input.png output.png
    python examples/input_output.py cifar10-crystal.safetensors examples/assets/frog.png input.png output.png
    python examples/input_output.py checkers-crystal.safetensors "<position>" input.png output.png
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import neuralcrystal as nc
from neuralcrystal import detectors, pictures

crystal_path, what, in_path, out_path = sys.argv[1:5]
crystal = nc.load(crystal_path)
problem = crystal.m.get("problem")

BG, INK, INK2, EDGE = (0, 0, 0), (223, 230, 234), (159, 176, 186), (58, 70, 80)
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Michroma-Regular.ttf")


def font(px):
    try:
        return ImageFont.truetype(FONT, px)
    except OSError:
        return ImageFont.load_default()


def framed(img, pad=18):
    out = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), BG); out.paste(img, (pad, pad))
    ImageDraw.Draw(out).rectangle([0, 0, out.width - 1, out.height - 1], outline=EDGE, width=2)
    return out


def labeled_squares(E, names, chosen, size, rows=None):
    """The exit face (as the camera sees it) with each readout square outlined and named."""
    face = Image.fromarray(pictures.exit_light(crystal, E, boxes=False, size=size))
    g = ImageDraw.Draw(face)
    N = crystal.N; w = max(1, int(round(N * crystal.win_frac))); off = (N - w) // 2; k = size / w
    for d, (x, y, bw, bh) in enumerate(detectors.rects(crystal)):
        x, y = x - off, y - off
        if crystal.sensor_turned: x, y = w - x - bw, w - y - bh                 # the relay turns the picture over on its way to the camera
        box = [x * k, y * k, (x + bw) * k - 1, (y + bh) * k - 1]
        g.rectangle(box, outline=pictures.CHOSEN if d == chosen else pictures.MARK, width=3 if d == chosen else 2)
        g.text(((box[0] + box[2]) / 2, box[1] - 5), names[d], fill=INK, font=font(12 if len(names[d]) < 4 else 10), anchor="mb")
    return face


if problem == "mnist":
    from neuralcrystal import mnist
    img = np.array(Image.open(what).convert("L"))
    digit, scores = mnist.classify(crystal, img)
    frames = mnist.frame(img)
    framed(Image.fromarray(pictures.frame(frames[0], scale=14))).save(in_path)
    E, _ = crystal.run(frames)
    framed(labeled_squares(E, [str(d) for d in range(10)], int(digit[0]), 600)).save(out_path)
    print(f"digit {int(digit[0])}")

elif problem == "cifar10":
    from neuralcrystal import cifar10
    photo = np.array(Image.open(what).convert("RGB"))
    cls, scores = cifar10.classify(crystal, photo)
    frames = cifar10.frame(crystal, photo)
    # the input: the photo, and the picture on the mirrors, side by side and labeled
    left = Image.fromarray(np.kron(photo, np.ones((16, 16, 1), np.uint8)))          # 512 × 512
    right = Image.fromarray(pictures.frame(frames[0], scale=2))                         # 512 × 512
    pan = Image.new("RGB", (512 * 2 + 40, 512 + 40), BG); pan.paste(left, (0, 40)); pan.paste(right, (552, 40))
    g = ImageDraw.Draw(pan)
    g.text((256, 18), "the photo, 32 × 32", fill=INK, font=font(14), anchor="mm")
    g.text((552 + 256, 18), "on the mirrors, 256 × 256", fill=INK, font=font(14), anchor="mm")
    framed(pan).save(in_path)
    E, _ = crystal.run(frames)
    L = cifar10.LABELS
    names = [n + " +" for n in L] + [n + " −" for n in L]
    framed(labeled_squares(E, names, int(cls[0]), 900)).save(out_path)
    print(f"class {L[int(cls[0])]}")

elif problem == "checkers":
    from neuralcrystal import checkers as ck
    pos = ck.parse(what)
    move, scores = ck.best_move(crystal, pos)
    frame = ck.frames([pos])[0].numpy()
    S, T = 5, 12                                                                      # screen pixels a picture pixel; a board cell is 12 picture pixels
    sq, _ = ck.view(pos)
    def cell(s): r = s >> 2; c = 2 * (s & 3) + (1 - (r & 1)); return c * T, r * T
    marks = []
    for want, label in ((1, "crystal's piece"), (2, "crystal's king"), (-1, "opponent's piece"), (-2, "opponent's king"), (0, "empty square")):   # the crystal is always the side to move
        s = next((i for i in range(32) if sq[i] == want), None)
        if s is not None: marks.append((cell(s), label))
    LF = font(13)
    LM = int(max(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(t, font=LF) for _, t in marks)) + 70
    img = Image.new("RGB", (LM + 96 * S, 96 * S), BG); img.paste(Image.fromarray(pictures.frame(frame, scale=S)), (LM, 0))
    g = ImageDraw.Draw(img)
    marks.sort(key=lambda m: m[0][1])                                                # top to bottom, so the leader lines don't cross
    ys = [(y + T / 2) * S for (x, y), _ in marks]
    for i in range(1, len(ys)): ys[i] = max(ys[i], ys[i - 1] + 44)                  # each label at its piece's height, never crowded
    for ((x, y), label), ly in zip(marks, ys):
        r = [LM + x * S, y * S, LM + (x + T) * S - 1, (y + T) * S - 1]
        g.rectangle(r, outline=INK, width=2)
        g.text((0, ly), label, fill=INK, font=LF, anchor="lm")
        g.line([g.textlength(label, font=LF) + 8, ly, r[0] - 2, (r[1] + r[3]) / 2], fill=INK, width=1)
    framed(img).save(in_path)
    # the output: the 32 × 32 move map, legal moves outlined, the crystal's in white
    E, _ = crystal.run(ck.frames([pos]))
    mm = ck.move_map(crystal, E)[0].cpu().double().numpy(); mm = (mm / mm.max()) ** 0.5
    C, Lm, Tm = 16, 70, 50
    out = Image.new("RGB", (Lm + 32 * C + 20, Tm + 32 * C + 40), BG)
    out.paste(Image.fromarray(np.kron(pictures.light(mm, gamma=1.0), np.ones((C, C, 1), np.uint8))), (Lm, Tm))
    g = ImageDraw.Draw(out)
    for m, c in ck.view_moves(pos):
        f, t = ck.cell_move(c); x, y = Lm + t * C, Tm + f * C
        g.rectangle([x, y, x + C - 1, y + C - 1], outline=pictures.CHOSEN if m == move else pictures.MARK, width=2)
    for k in (0, 7, 15, 23, 31):
        g.text((Lm + (k + 0.5) * C, Tm + 32 * C + 14), str(k + 1), fill=INK2, font=font(10), anchor="mm")
        g.text((Lm - 10, Tm + (k + 0.5) * C), str(k + 1), fill=INK2, font=font(10), anchor="rm")
    g.text((Lm + 16 * C, 18), "to square", fill=INK, font=font(14), anchor="mm")
    g.text((0, Tm - 18), "from square", fill=INK, font=font(14), anchor="lm")
    framed(out).save(out_path)
    print(f"plays {ck.uci(move)}")
else:
    raise SystemExit(f"{problem}: use chess_input_output.py for chess")
print(f"wrote {in_path} and {out_path}")
