"""The chess crystal's input and output, drawn large and labeled.

    python examples/chess_input_output.py chess-crystal.safetensors ["<FEN>"] [board.png] [move-map.png]

board.png: the 192 × 192 picture the micromirror array shows the crystal, enlarged.
move-map.png: the 64 × 64 move map the light lands on. Row = the square a piece moves from, column = the square it moves to (both as
the crystal sees the board: always from the side to move, so white's positions are mirrored). Legal moves are outlined in blue and the
crystal's move in orange. A piece can't move to its own square, so the diagonal is free; the crystal uses three stretches of it to read
out how the game is going: win, draw and loss for the side to move.
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import neuralcrystal as nc
from neuralcrystal import chess as ch, pictures

crystal_path = sys.argv[1]
fen = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
board_path = sys.argv[3] if len(sys.argv) > 3 else "board.png"
map_path = sys.argv[4] if len(sys.argv) > 4 else "move-map.png"

crystal = nc.load(crystal_path)
pos = ch.parse(fen)
pick = ch.best_move(crystal, pos)
w, d, l = (float(x) for x in pick.wdl)
print(f"the crystal plays {ch.san(pos, pick.move)}; win / draw / loss {100 * w:.0f} / {100 * d:.0f} / {100 * l:.0f} %")


# the colors of neuralcrystal.com; the background is black, like the board's dark mirrors, so picture and labels share one ground
BG, INK, INK2, INK3, EDGE = (0, 0, 0), (223, 230, 234), (159, 176, 186), (111, 128, 138), (58, 70, 80)
ACCENT, WIRE = (230, 102, 74), (40, 178, 146)
# the label font of neuralcrystal.com: Michroma (SIL Open Font License, examples/fonts/OFL.txt)
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Michroma-Regular.ttf")


def font(px):
    try:
        return ImageFont.truetype(FONT, px)
    except OSError:
        return ImageFont.load_default()


def framed(img, pad=18):
    """The picture on the same black ground, inside a thin border."""
    out = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), BG); out.paste(img, (pad, pad))
    ImageDraw.Draw(out).rectangle([0, 0, out.width - 1, out.height - 1], outline=EDGE, width=2)
    return out


# the input: the board picture, enlarged, with its markers labeled. The art is 96 × 96 (each pixel doubled on the mirrors);
# along its left edge sit the turn lights, the castling flags and the quartz mark showing which side the crystal plays.
frames = ch.frames([pos])
art = ch.render96(ch.view(pos, ch.look_of(crystal).side)[0], ch.look_of(crystal))
S = 5                                                                # 5 × 5 screen pixels a picture pixel
black_to_move = pos.turn == "b"
marks = [  # (rect in art pixels, label)
    (ch.LAMP["crystalT"] if black_to_move else ch.LAMP["crystalB"], "the crystal's side (black)"),
    (ch.LAMP["castle"]["q"], "castling: black queenside"),
    (ch.LAMP["castle"]["k"], "castling: black kingside"),
    (ch.LAMP["turnB"], "turn light: black to move"),
    (ch.LAMP["turnW"], "turn light: white to move"),
    (ch.LAMP["castle"]["Q"], "castling: white queenside"),
    (ch.LAMP["castle"]["K"], "castling: white kingside"),
]
LF = font(13)
LM = int(max(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(t, font=LF) for _, t in marks)) + 70
bimg = Image.new("RGB", (LM + 96 * S, 96 * S), BG)
bimg.paste(Image.fromarray(pictures.frame(art, scale=S)), (LM, 0))                    # lit mirrors in the laser's red
bd = ImageDraw.Draw(bimg)
ys = np.linspace(18, 96 * S - 18, len(marks))                        # label rows, spread down the left margin
for ((mx, my, mw, mh), label), ly in zip(marks, ys):
    rx0, ry0, rx1, ry1 = LM + mx * S - 1, my * S - 1, LM + (mx + mw) * S, (my + mh) * S
    bd.rectangle([rx0, ry0, rx1, ry1], outline=INK, width=2)
    bd.text((0, ly), label, fill=INK, font=LF, anchor="lm")
    tx = bd.textlength(label, font=LF) + 8
    bd.line([tx, ly, rx0 - 2, (ry0 + ry1) / 2], fill=INK, width=1)
framed(bimg).save(board_path)

# the output: the move map, each cell C × C, square-root brightness so faint light shows
E, _ = crystal.run(frames)
mm = ch.move_map(crystal, E)[0].cpu().double().numpy()
mm = (mm / mm.max()) ** 0.5
C, L, T = 9, 80, 50                                                  # cell size, left and top margins for the labels
size = 64 * C
img = Image.new("RGB", (L + size + 200, T + size + 40), BG)
img.paste(Image.fromarray(np.kron(pictures.light(mm, gamma=1.0), np.ones((C, C, 1), np.uint8))), (L, T))
g = ImageDraw.Draw(img)


def box(r, c, color, wd=1):
    x, y = L + c * C, T + r * C
    g.rectangle([x, y, x + C - 1, y + C - 1], outline=color, width=wd)


for cell, m in ch.legal_cells(pos).items():
    f, t = ch.cell_move(cell); box(f, t, WIRE)
f, t = next(ch.cell_move(c) for c, m in ch.legal_cells(pos).items() if m == pick.move); box(f, t, pictures.CHOSEN, 2)

# the value readout: three runs of the diagonal
runs = {"win": ((0, 15), WIRE), "draw": ((24, 39), INK2), "loss": ((48, 63), ACCENT)}
share = {"win": w, "draw": d, "loss": l}
for name, ((a, b), color) in runs.items():
    for sq in range(a, b + 1): box(sq, sq, color, 2)              # each cell of the run, on the diagonal
    xm, ym = L + (b + 1) * C, T + ((a + b + 1) / 2) * C
    g.line([xm + 4, ym, xm + 40, ym], fill=color, width=2)
    g.text((xm + 46, ym), f"{name} {100 * share[name]:.0f} %", fill=INK, font=font(14), anchor="lm")

# axes: rank boundaries every 8 squares
for k in range(9):
    g.line([L + 8 * k * C, T + size, L + 8 * k * C, T + size + 6], fill=INK3)
    g.line([L - 6, T + 8 * k * C, L, T + 8 * k * C], fill=INK3)
for k in range(8):
    g.text((L + (8 * k + 4) * C, T + size + 16), f"rank {k + 1}", fill=INK2, font=font(10), anchor="mm")
    g.text((L - 10, T + (8 * k + 4) * C), f"rank {k + 1}", fill=INK2, font=font(10), anchor="rm")
g.text((L + size / 2, 18), "to square", fill=INK, font=font(14), anchor="mm")
g.text((0, T - 18), "from square", fill=INK, font=font(14), anchor="lm")
framed(img).save(map_path)
print(f"wrote {board_path} and {map_path}")
