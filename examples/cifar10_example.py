"""Classify a CIFAR-10 photo with the CIFAR-10 crystal, and draw every picture the light makes on the way.

    python examples/cifar10_example.py cifar10-crystal.safetensors examples/assets/frog.png [out.png]

The image is a 32 × 32 colour photo. Prints the class (and the class with the mirrored photo's scores added) and writes a picture:
the DMD frame | the camera's picture after passes 1, 2 and 3 (what the DMD shows the next pass) | the light on the exit face, the
twenty squares outlined (ten differential pairs), the answer's plus square in orange.
"""
import sys
import numpy as np
from PIL import Image

import neuralcrystal as nc
from neuralcrystal import cifar10, pictures

crystal_path, image_path = sys.argv[1], sys.argv[2]
out_path = sys.argv[3] if len(sys.argv) > 3 else "cifar10-out.png"

crystal = nc.load(crystal_path)
photo = np.asarray(Image.open(image_path).convert("RGB"))         # uint8 [32, 32, 3]

cls, scores = cifar10.classify(crystal, photo)
cls_f, _ = cifar10.classify(crystal, photo, flip=True)
print(f"class: {cifar10.LABELS[int(cls[0])]}   (with the mirrored photo too: {cifar10.LABELS[int(cls_f[0])]})")
for k in scores[0].argsort(descending=True)[:3].tolist(): print(f"  {cifar10.LABELS[k]:<11} {float(scores[0, k]):+.3f}")

frames = cifar10.frame(crystal, photo)
E, _ = crystal.run(frames)                                         # crystal.hidden_pictures: the camera's picture after each pass but the last
pics = [pictures.frame(frames[0], scale=2)] + [pictures.frame(h[0], scale=2) for h in crystal.hidden_pictures]
pictures.save(pictures.strip(pics + [pictures.exit_light(crystal, E, highlight=int(cls[0]))]), out_path)
print(f"wrote {out_path}")
