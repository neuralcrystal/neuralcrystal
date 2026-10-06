"""Read a handwritten digit with the MNIST crystal, and draw what the light did.

    python examples/mnist_example.py mnist-crystal.safetensors examples/assets/digit-4.png [out.png]

The image is a 28 × 28 MNIST-style digit (white ink on black). Prints the digit and the light on each of the ten squares, and writes
a picture: the DMD frame going in | the light on the crystal's exit face, the ten squares outlined, the answer's in orange.
"""
import sys
import numpy as np
from PIL import Image

import neuralcrystal as nc
from neuralcrystal import mnist, pictures

crystal_path, image_path = sys.argv[1], sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else "mnist-out.png"

crystal = nc.load(crystal_path)                                   # fp32, -16bit or -8bit: all three load the same way
img = np.asarray(Image.open(image_path).convert("L"))              # uint8 [28, 28]

digit, scores = mnist.classify(crystal, img)
share = scores[0] / scores[0].sum()
print(f"digit: {int(digit[0])}")
for d in range(10): print(f"  {d}  {'#' * int(round(60 * float(share[d])))} {100 * float(share[d]):.1f} %")

# the pictures: run once more by hand to keep the exit field
frames = mnist.frame(img)
E, _ = crystal.run(frames)
pictures.save(pictures.strip([pictures.frame(frames[0], scale=16), pictures.exit_light(crystal, E, highlight=int(digit[0]))]), out_path)
print(f"wrote {out_path}")
