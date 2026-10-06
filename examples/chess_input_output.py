"""The chess crystal's input and output, drawn large and labeled.

    python examples/chess_input_output.py chess-crystal.safetensors ["<FEN>"] [board.png] [move-map.png]

board.png: the 192 × 192 picture the micromirror array shows the crystal, enlarged.
move-map.png: the 64 × 64 move map the light lands on. Row = the square a piece moves from, column = the square it moves to (both as
the crystal sees the board: always from the side to move, so white's positions are mirrored). Legal moves are outlined in blue and the
crystal's move in orange. A piece can't move to its own square, so the diagonal is free; the crystal uses three stretches of it to read
out how the game is going: win, draw and loss for the side to move.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import neuralcrystal as nc
from neuralcrystal import chess as ch

crystal_path = sys.argv[1]
fen = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
board_path = sys.argv[3] if len(sys.argv) > 3 else "board.png"
map_path = sys.argv[4] if len(sys.argv) > 4 else "move-map.png"

crystal = nc.load(crystal_path)
pos = ch.parse(fen)
pick = ch.best_move(crystal, pos)
w, d, l = (float(x) for x in pick.wdl)
print(f"the crystal plays {ch.san(pos, pick.move)}; win / draw / loss {100 * w:.0f} / {100 * d:.0f} / {100 * l:.0f} %")


def font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


# the input: the board picture, enlarged, with its markers labeled. The art is 96 × 96 (each pixel doubled on the mirrors);
# along its left edge sit the turn lights, the castling flags and the quartz mark showing which side the crystal plays.
frames = ch.frames([pos])
art = ch.render96(ch.view(pos, ch.look_of(crystal).side)[0], ch.look_of(crystal))
S, LM = 5, 330                                                       # 5 × 5 screen pixels a picture pixel, room for labels on the left
bimg = Image.new("RGB", (LM + 96 * S + 20, 96 * S + 20), (16, 16, 16))
bimg.paste(Image.fromarray(np.kron((art * 255).astype(np.uint8), np.ones((S, S), np.uint8))).convert("RGB"), (LM, 10))
bd = ImageDraw.Draw(bimg)
black_to_move = pos.turn == "b"
marks = [  # (rect in art pixels, label, color)
    (ch.LAMP["crystalT"] if black_to_move else ch.LAMP["crystalB"], "the crystal's side (it plays black)", (120, 200, 255)),
    (ch.LAMP["castle"]["q"], "castling: black queenside", (255, 190, 80)),
    (ch.LAMP["castle"]["k"], "castling: black kingside", (255, 190, 80)),
    (ch.LAMP["turnB"], "turn light: black to move", (120, 230, 140)),
    (ch.LAMP["turnW"], "turn light: white to move", (120, 230, 140)),
    (ch.LAMP["castle"]["Q"], "castling: white queenside", (255, 190, 80)),
    (ch.LAMP["castle"]["K"], "castling: white kingside", (255, 190, 80)),
]
ys = np.linspace(28, 96 * S - 18, len(marks))                        # label rows, spread down the left margin
for (x, y, w, h), (label, color), ly in zip([m[0] for m in marks], [(m[1], m[2]) for m in marks], ys):
    rx0, ry0, rx1, ry1 = LM + x * S - 1, 10 + y * S - 1, LM + (x + w) * S, 10 + (y + h) * S
    bd.rectangle([rx0, ry0, rx1, ry1], outline=color, width=2)
    bd.text((12, ly), label, fill=color, font=font(17), anchor="lm")
    tx = 12 + bd.textlength(label, font=font(17)) + 8
    bd.line([tx, ly, rx0 - 2, (ry0 + ry1) / 2], fill=color, width=1)
bimg.save(board_path)

# the output: the move map, each cell C × C, square-root brightness so faint light shows
E, _ = crystal.run(frames)
mm = ch.move_map(crystal, E)[0].cpu().double().numpy()
mm = (mm / mm.max()) ** 0.5
C, L, T = 9, 70, 60                                                  # cell size, left and top margins for the labels
size = 64 * C
img = Image.new("RGB", (L + size + 190, T + size + 50), (16, 16, 16))
cells = np.kron((mm * 255).astype(np.uint8), np.ones((C, C), np.uint8))
img.paste(Image.fromarray(np.stack([cells] * 3, -1)), (L, T))
g = ImageDraw.Draw(img)


def box(r, c, color, wd=1):
    x, y = L + c * C, T + r * C
    g.rectangle([x, y, x + C - 1, y + C - 1], outline=color, width=wd)


for cell, m in ch.legal_cells(pos).items():
    f, t = ch.cell_move(cell); box(f, t, (90, 150, 255))
f, t = next(ch.cell_move(c) for c, m in ch.legal_cells(pos).items() if m == pick.move); box(f, t, (255, 170, 0), 2)

# the value readout: three runs of the diagonal
runs = {"win": ((0, 15), (80, 220, 120)), "draw": ((24, 39), (230, 230, 230)), "loss": ((48, 63), (240, 90, 90))}
share = {"win": w, "draw": d, "loss": l}
for name, ((a, b), color) in runs.items():
    for sq in range(a, b + 1): box(sq, sq, color, 2)              # each cell of the run, on the diagonal
    xm, ym = L + (b + 1) * C, T + ((a + b + 1) / 2) * C
    g.line([xm + 4, ym, xm + 40, ym], fill=color, width=2)
    g.text((xm + 46, ym), f"{name} {100 * share[name]:.0f} %", fill=color, font=font(18), anchor="lm")

# axes: rank boundaries every 8 squares
for k in range(9):
    g.line([L + 8 * k * C, T + size, L + 8 * k * C, T + size + 6], fill=(140, 140, 140))
    g.line([L - 6, T + 8 * k * C, L, T + 8 * k * C], fill=(140, 140, 140))
for k in range(8):
    g.text((L + (8 * k + 4) * C, T + size + 16), f"rank {k + 1}", fill=(170, 170, 170), font=font(13), anchor="mm")
    g.text((L - 10, T + (8 * k + 4) * C), f"rank {k + 1}", fill=(170, 170, 170), font=font(13), anchor="rm")
g.text((L + size / 2, 22), "to square", fill=(220, 220, 220), font=font(18), anchor="mm")
g.text((12, T - 22), "from square", fill=(220, 220, 220), font=font(18), anchor="lm")
img.save(map_path)
print(f"wrote {board_path} and {map_path}")
