"""Ask the chess crystal for a move (and, optionally, let it think two moves ahead), and draw what the light did.

    python examples/chess_example.py chess-crystal.safetensors ["<FEN>"] [out.png] [--think]

Prints the crystal's move, its five brightest moves and its win/draw/loss reading for the side to move. Writes a picture: the
192 × 192 DMD frame (always shown from the side to move, so white's positions are mirrored) | the camera's picture after passes 1, 2
and 3 | the 64 × 64 move map (row = from-square, column = to-square, as the crystal sees the board), the legal moves outlined in blue
and the crystal's move in orange. With --think the crystal also reads the boards after its best three moves and the opponent's best
five replies, and plays the line that holds up best.
"""
import sys
import numpy as np

import neuralcrystal as nc
from neuralcrystal import chess as ch, pictures

args = [a for a in sys.argv[1:] if not a.startswith("--")]; think = (3, 5) if "--think" in sys.argv else None
crystal_path = args[0]
fen = args[1] if len(args) > 1 and args[1] else "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
out_path = args[2] if len(args) > 2 else "chess-out.png"

crystal = nc.load(crystal_path)
pos = ch.parse(fen)
pick = ch.best_move(crystal, pos, think=think)
print(f"{'white' if pos.turn == 'w' else 'black'} to move: the crystal plays {ch.san(pos, pick.move)} ({ch.uci(pick.move)})")
print("  win / draw / loss for the side to move: " + " / ".join(f"{100 * float(x):.0f} %" for x in pick.wdl))
for m, v in pick.scores[:5]: print(f"  {ch.san(pos, m):<7} {'#' * int(round(40 * v / pick.scores[0][1]))}")
if think: print(f"  thought about {pick.think['cand']}, expected scores {[round(x, 2) for x in pick.think['es']]}")

frames = ch.frames([pos])
E, _ = crystal.run(frames)
mm = pictures.light(ch.move_map(crystal, E)[0], scale=6)           # 64 × 64 cells → 384 × 384
for cell, m in ch.legal_cells(pos).items():                        # outline every legal move's cell; the crystal's in orange
    f, t = ch.cell_move(cell); y, x = 6 * f, 6 * t; col = (255, 170, 0) if m == pick.move else (80, 140, 255)
    mm[y:y + 6, x] = col; mm[y:y + 6, x + 5] = col; mm[y, x:x + 6] = col; mm[y + 5, x:x + 6] = col
pics = [pictures.frame(frames[0], scale=2)] + [pictures.frame(h[0], scale=2) for h in crystal.hidden_pictures] + [mm]
pictures.save(pictures.strip(pics), out_path)
print(f"wrote {out_path}")
