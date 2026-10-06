"""Crystal files: the v2 safetensors container (fp32 maps), and its packed form (cropped, phase wrapped to 16 or 8 bits).

A v2 file is a safetensors container whose __metadata__["neuralcrystal"] holds the crystal's model JSON (geometry, imager, quad
hand-off, readout) and whose tensors are the learned surfaces: plane.<i>.phi for section 0 and plane.<i>.phi.<k> for section k of a
quad crystal, float32 [n, n] phase in radians.

A packed file (the web form, packages/engine/src/crystal/pack.ts) is the same container with each map cropped to the square that
holds its nonzero samples and its phase wrapped to one turn and stored as U16 or U8; __metadata__["nc.pack"] says where each crop
sits. exp(iφ) is all the light sees, so wrapping changes nothing; quantising moves each sample by at most half a step
(16 bits: 0.00005 rad, 8 bits: 0.012 rad).
"""
import json, math
import numpy as np

TAU = 2 * math.pi
FORMAT_V2 = "neuralcrystal.model.v2"
PACK_FORMAT = "nc.pack.v1"
_DT = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32, "I8": np.int8, "U8": np.uint8, "U16": np.uint16, "BOOL": np.bool_}
_DT_NAME = {np.dtype(v): k for k, v in _DT.items()}


def st_read(path):
    """A safetensors file → (metadata: dict of str, tensors: dict name → numpy array)."""
    with open(path, "rb") as f: buf = f.read()
    n = int.from_bytes(buf[:8], "little"); hdr = json.loads(buf[8:8 + n].decode("utf8")); base = 8 + n
    meta = hdr.pop("__metadata__", {}) or {}; out = {}
    for name, t in hdr.items():
        a, b = t["data_offsets"]; dt = np.dtype(_DT[t["dtype"]])
        out[name] = np.frombuffer(buf, dtype=dt, count=(b - a) // dt.itemsize, offset=base + a).reshape(t["shape"])
    return meta, out


def st_write(path, meta, tensors):
    """Write a safetensors file (tensors in name order, header padded to a multiple of 8). Metadata values must be strings."""
    hdr = {"__metadata__": {k: str(v) for k, v in meta.items()}}; off = 0; blobs = []
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name]); blob = a.tobytes()
        hdr[name] = {"dtype": _DT_NAME[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + len(blob)]}; off += len(blob); blobs.append(blob)
    h = json.dumps(hdr, separators=(",", ":")).encode("utf8"); h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(len(h).to_bytes(8, "little")); f.write(h)
        for b in blobs: f.write(b)


def unpack_map(q, p, bits):
    """A packed map back to its full n × n phase map: radians in (-π, π], zeros outside its window (pack.ts unpackMap)."""
    Q = 65536 if bits == 16 else 256
    phi = q.astype(np.float64) / Q * TAU
    phi = np.where(phi > math.pi, phi - TAU, phi)
    out = np.zeros((p["n"], p["n"]), dtype=np.float32)
    out[p["y0"]:p["y0"] + p["w"], p["x0"]:p["x0"] + p["w"]] = phi.astype(np.float32)
    return out


def read_model(path):
    """The model dict from a v2 or packed file, with each learned plane's maps attached as numpy arrays (phi, and phis for a quad)."""
    meta, tensors = st_read(path)
    m = json.loads(meta["neuralcrystal"])
    if m.get("format") != FORMAT_V2 and meta.get("format") != FORMAT_V2: raise ValueError("not a NeuralCrystal v2 crystal file")
    if "nc.pack" in meta:
        pk = json.loads(meta["nc.pack"])
        if pk.get("format") != PACK_FORMAT: raise ValueError(f"unknown crystal packing {pk.get('format')}")
        tensors = {k: unpack_map(v, pk["maps"][k], int(pk["bits"])) for k, v in tensors.items() if k in pk["maps"]}
    for i, p in enumerate(m["planes"]):
        if f"plane.{i}.phi" in tensors: p["phi"] = tensors[f"plane.{i}.phi"]
        ks = [k for k in tensors if k.startswith(f"plane.{i}.phi.")]
        if ks:
            ph = [None] * (max(int(k.split(".")[-1]) for k in ks) + 1)
            for k in ks: ph[int(k.split(".")[-1])] = tensors[k]
            if p.get("phi") is not None: ph[0] = p["phi"]
            p["phis"] = ph
    return m


def _window(m):
    """The smallest square about the map's centre (a multiple of 8, within the map) holding every nonzero sample (pack.ts windowOf)."""
    n = m.shape[0]; c = n / 2
    ys, xs = np.nonzero(m)
    r = 0.0 if len(xs) == 0 else float(max(np.abs(xs + 0.5 - c).max(), np.abs(ys + 0.5 - c).max()))
    w = min(n, int(math.ceil(2 * r / 8)) * 8)
    if w <= 0: w = min(n, 8)
    x0 = max(0, int(math.floor(c - w / 2 + 0.5)))
    return x0, x0, w


def pack(src, dst, bits):
    """Write the packed form of a v2 file at 16 or 8 bits (pack.ts packCrystal, sample for sample)."""
    if bits not in (8, 16): raise ValueError("bits must be 8 or 16")
    meta, tensors = st_read(src)
    if "nc.pack" in meta: raise ValueError(f"{src} is already packed")
    Q = 65536 if bits == 16 else 256; dt = np.uint16 if bits == 16 else np.uint8
    out = {}; maps = {}
    for name, a in tensors.items():
        if not name.startswith("plane."): continue                     # anything else (training state) is not part of a crystal
        a = np.asarray(a, dtype=np.float32); n = a.shape[0]
        x0, y0, w = _window(a)
        phi = a[y0:y0 + w, x0:x0 + w].astype(np.float64)
        t = phi / TAU - np.floor(phi / TAU)
        out[name] = (np.floor(t * Q + 0.5).astype(np.int64) % Q).astype(dt)   # Math.round: halves go up
        maps[name] = {"n": n, "x0": x0, "y0": y0, "w": w}
    st_write(dst, {"neuralcrystal": meta["neuralcrystal"], "nc.pack": json.dumps({"format": PACK_FORMAT, "bits": bits, "maps": maps}, separators=(",", ":"))}, out)
