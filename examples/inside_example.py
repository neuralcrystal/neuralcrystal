"""Look inside a crystal: the light arriving at every surface of every pass, the camera's pictures between passes, and the answer.

    python examples/inside_example.py cifar10-crystal.safetensors examples/assets/frog.png [inside.png]
    python examples/inside_example.py checkers-crystal.safetensors "bb.bb...b.wb.b.b....w.ww.w.ww.ww w - 0" [inside.png]
    python examples/inside_example.py chess-crystal.safetensors "<FEN>" [inside.png]

Writes one picture, one row per pass: what the DMD launched, the light arriving at each of the section's surfaces, and the light at
its exit face. For checkers and chess, a last row adds the move map (32 × 32 / 64 × 64) (row = from-square, column = to-square), enlarged.
"""
import sys
import numpy as np
from PIL import Image

import neuralcrystal as nc
from neuralcrystal import pictures

crystal_path, what = sys.argv[1], sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else "inside.png"
crystal = nc.load(crystal_path)

problem = crystal.m.get("problem")
if problem == "checkers":
    from neuralcrystal import checkers as ck
    frames = ck.frames([ck.parse(what)])
elif problem in ("chess", "chesswin"):
    from neuralcrystal import chess as ch
    frames = ch.frames([ch.parse(what)])
elif problem == "cifar10":
    from neuralcrystal import cifar10
    frames = cifar10.frame(crystal, np.array(Image.open(what).convert("RGB")))
else:
    from neuralcrystal import mnist
    frames = mnist.frame(np.array(Image.open(what).convert("L")))

E, steps = crystal.trace(frames, n=128)                         # the exit field, and the light at every surface of every pass
launched = [frames[0]] + [h[0] for h in crystal.hidden_pictures]   # what the DMD showed each pass: the input, then the camera's pictures
T = steps[0]["light"].shape[-1]                                 # every tile drawn at the traced light's size
rows = []
for k in range(crystal.hops):
    row = [pictures.frame(launched[k], scale=max(1, round(T / launched[k].shape[-1])))]
    row += [pictures.light(s["light"][0]) for s in steps if s["pass"] == k]
    rows.append(row)
    print(f"pass {k + 1}: launched a {launched[k].shape[-1]} × {launched[k].shape[-1]} picture, light traced at {len(row) - 2} surfaces and the exit face")
if problem == "checkers":
    rows.append([pictures.light(ck.move_map(crystal, E)[0], scale=max(1, T // 32))])
elif problem in ("chess", "chesswin"):
    rows.append([pictures.light(ch.move_map(crystal, E)[0], scale=max(1, T // 64))])
n_surf = len(rows[0]) - 2
cols = ["DMD in"] + [f"surface {i + 1}" for i in range(n_surf)] + ["exit face"]
labels = [f"pass {k + 1}" for k in range(crystal.hops)] + (["move map"] if len(rows) > crystal.hops else [])
pictures.save(pictures.grid(rows, row_labels=labels, col_labels=cols), out_path)
print(f"wrote {out_path}")
