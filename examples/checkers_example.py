"""Ask the checkers crystal for a move, and draw what the light did.

    python examples/checkers_example.py checkers-crystal.safetensors ["bb.bb...b.wb.b.b....w.ww.w.ww.ww w - 0"] [out.png]

A position is 32 characters, one per playable square in draughts order (b/w = black/white man, B/W = king, . = empty), then the side
to move, the square of a piece that must keep jumping (or -) and the moves since the last capture. Prints the crystal's move and the light
on every legal move, and writes a picture: the 96 × 96 DMD frame | the camera's picture after passes 1, 2 and 3 | the exit face's
move map (a 32 × 32 grid: row = from-square, column = to-square, both as the side to move sees the board), the legal moves outlined in
blue and the crystal's move in orange.
"""
import sys
import numpy as np

import neuralcrystal as nc
from neuralcrystal import checkers as ck, pictures

crystal_path = sys.argv[1]
position = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "bbbbbbbbbbbb........wwwwwwwwwwww b - 0"   # the opening position
out_path = sys.argv[3] if len(sys.argv) > 3 else "checkers-out.png"

crystal = nc.load(crystal_path)
pos = ck.parse(position)
move, scores = ck.best_move(crystal, pos)                          # the brightest legal move; temp/top_k sample for variety
total = sum(v for _, v in scores)
print(f"{'black' if pos.turn > 0 else 'white'} to move: the crystal plays {ck.uci(move)}")
for m, v in scores: print(f"  {ck.uci(m):<6} {'#' * int(round(50 * v / scores[0][1]))} {100 * v / total:.1f} %")

frames = ck.frames([pos])
E, _ = crystal.run(frames)
face = pictures.exit_light(crystal, E, boxes=False, size=0)        # the window at one pixel a sample
lo, side = ck.bands(crystal.N, crystal.win_frac); x0 = (crystal.N - face.shape[0]) // 2
for m, cell in ck.view_moves(pos):                                 # outline each legal move's cell; the chosen one in orange
    f, t = ck.cell_move(cell); y, x = lo[f] - x0, lo[t] - x0; col = (255, 170, 0) if m == move else (80, 140, 255)
    face[y:y + side, x] = col; face[y:y + side, x + side - 1] = col; face[y, x:x + side] = col; face[y + side - 1, x:x + side] = col
pics = [pictures.frame(frames[0], scale=4)] + [pictures.frame(h[0], scale=4) for h in crystal.hidden_pictures] + [face[::max(1, face.shape[0] // 384)][:, ::max(1, face.shape[0] // 384)]]
pictures.save(pictures.strip(pics), out_path)
print(f"wrote {out_path}")
