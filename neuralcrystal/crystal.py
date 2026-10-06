"""A NeuralCrystal crystal in PyTorch, forward only: load a crystal file, launch a batch of DMD frames into it, propagate the light
through every surface, and read the sensor out.

Scalar, coherent, one wavelength (617 nm) in fused silica: the band-limited angular spectrum between thin phase surfaces, the learned
surfaces as blocks of samples, the DMD frame as amplitude blocks in the extent box (pre-flipped 180°), and the readout as the box
binned to the frame. A quad crystal runs n passes through n sections of glass; between passes the camera's picture goes through the
imager (auto-exposure, black level, γ) and is dithered back onto the DMD at the mirrors' levels.

This is the light path of the trainer that made the published crystals, with the training machinery taken out:
no gradients, trainable gaps, pyramids or optimiser state. Features no published crystal uses are refused with an error rather than
approximated.
"""
import math
import numpy as np
import torch

from .files import read_model

TAU = 2 * math.pi


def rnd(x): return int(math.floor(x + 0.5))                                  # JavaScript's Math.round (halves go up), as the simulator that drew the geometry


def device_pick(name=None):
    if name: return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


class Crystal:
    def __init__(self, model, device=None, ext=None):
        self.m = model; self.dev = device_pick(device)
        ov = model.get("overlay")
        if isinstance(ov, dict) and ov.get("on"): raise NotImplementedError("crystals with the overlay path on are not supported")
        qd = model.get("quad") if isinstance(model.get("quad"), dict) else None
        if qd and qd.get("bnd"): raise NotImplementedError("quad.bnd (attention boundaries) is not supported")
        if qd and float(qd.get("ghost", 0) or 0): raise NotImplementedError("quad.ghost is not supported")

        self.N = int(model["N"]); self.dx = float(model["dx_um"]); self.lam = float(model.get("lam_um", 0.617)); self.nmed = float(model.get("n_medium", 1.4607))
        self.in_gap = float(model.get("inGap_mm", 0)); self.ext = float(ext if ext is not None else model.get("ext", 50))
        nl = model.get("nl") or {}
        self.nl = dict(t0=nl.get("t0", 0.67), tinf=nl.get("tinf", 0.86), knee=nl.get("knee", 0.45), phi=nl.get("phi", 0.08))
        self.nl_abs = bool(nl.get("abs")); self.nl_fsat = float(nl.get("fsat", 1)); self.nl_tau = float(nl.get("tau_s", 100e-12))
        self.nl_ppf = max(1.0, float(nl.get("ppf", 1))); self.nl_plen = float(nl.get("pulse_len_s", 10e-6))
        self.nl_se = float(nl.get("sensor_e", 1000)); self.nl_qe = float(nl.get("sensor_qe", 0.6)); self.nl_wshare = float(nl.get("win_share", 1))
        self.nl_frac = None                                                    # the stack's throughput on the last pass (sets the absolute knee of the next)
        im = model.get("imager") or {}
        self.img = dict(g=im.get("g", 1), b=im.get("b", 0.05), gm=im.get("gm", 1), flip=im.get("flip", True) is not False, inv=bool(im.get("inv", False)), mode=im.get("mode", "fixed"), sen=im.get("sen", "ideal"))
        self.whole = bool(model.get("wholeGlass", False)); self.win_um = float(model.get("winUm") or self.N * self.dx)
        self.win_frac = min(1.0, self.win_um / (self.N * self.dx)) if self.whole else 1.0          # the window's share of the grid
        self.mirror = bool(model.get("solveMirror", False)) and not bool(model.get("sideBlack", False))   # reflecting sides
        self.absorb = bool(model.get("absorbRim", False)) and self.whole and self.win_frac < 1.0  # black sides: the field outside the window is lost after every hop
        self.sub_hop = float(model.get("subHopUm") or 0.0); self.zmax_split = bool(model.get("zMaxSplit", False))
        self.dmd_na = float(model.get("dmdNA", 0) or 0)                        # the DMD relay's numerical aperture (0 = no pupil)
        if self.absorb:
            ws = max(1, rnd(self.N * self.win_frac)); wx = rnd((self.N - ws) / 2)
            self._absorb_mask = torch.zeros(self.N, self.N); self._absorb_mask[wx:wx + ws, wx:wx + ws] = 1.0
        self.sens_pitch = 2.9; self.mirrors = int(model.get("dmdMirrors") or 0)

        # the quad crystal: `hops` passes through `nsec` sections of glass, the camera's picture of pass k dithered onto the DMD for pass k+1
        self.hops = max(1, int(qd.get("n", 4) or 4)) if qd else 1
        self.q_pass = [int(x) for x in (qd.get("passes") or [])] if qd else []
        self.nsec = self.hops
        if self.q_pass: self.nsec = max(self.q_pass) + 1; self.hops = len(self.q_pass)
        self.q_mir = max(1, int(qd.get("mir", 4) or 4)) if qd else 4
        self.q_levels = max(2, int((qd.get("levels") if qd else 0) or (self.q_mir * self.q_mir + 1)))   # the dither's amplitude levels
        gs = [float(v) for v in (qd.get("gains") or [])] if qd else []; self.q_gains = (gs + [1.0] * self.hops)[:max(0, self.hops - 1)]
        self.q_res = float(qd.get("res", 0) or 0) if qd else 0.0                # each section's own input added back onto its output (a residual)
        self.q_hidden = int(qd.get("hidden", 0) or 0) if qd else 0
        nm = qd.get("norm") if qd else None
        self.q_norm = {"t": float(nm.get("t", 0) or 0), "s": float(nm.get("s", 4) or 4)} if isinstance(nm, dict) else None   # auto-exposure from each picture's mean and spread
        self.q_band = float(qd.get("band", 0) or 0) if qd else 0.0; self.q_band_y0 = int(qd.get("bandY0", 80) or 80) if qd else 80
        self.q_unrot = bool(qd.get("unrot", False)) if qd else False             # the FPGA turns the picture back 180° before writing it

        self.planes = []
        for p in model["planes"]:
            q = dict(kind=p["kind"], param=float(p.get("param", 0) or 0), dist=float(p.get("dist_mm", 0) or 0), off=bool(p.get("off", False)), n=None, phi=None,
                     kinds=list(p["kinds"]) if isinstance(p.get("kinds"), list) else None, params=list(p["params"]) if isinstance(p.get("params"), list) else None, arrs=list(p["arrs"]) if isinstance(p.get("arrs"), list) else None)
            if self.learned_any(q) and p.get("phi") is not None:
                q["phi"] = torch.tensor(np.array(p["phi"], dtype=np.float32)); q["n"] = int(q["phi"].shape[0])
            if self.learned_any(q) and isinstance(p.get("phis"), list):
                ph = [None if a is None else torch.tensor(np.array(a, dtype=np.float32)) for a in p["phis"]]
                if ph and ph[0] is not None: q["phi"] = ph[0]
                elif q["phi"] is not None and ph: ph[0] = q["phi"]
                q["phis"] = ph; q["n"] = q["n"] or next(int(t.shape[0]) for t in ph if t is not None)
            a = p.get("arr")
            q["arr"] = dict(nx=int(a.get("nx", 1) or 1), ny=int(a.get("ny", 1) or 1), pitch=float(a.get("pitch_mm", a.get("pitch", 0.5)) or 0.5), cells=[(float(v) if (v is not None and float(v) > 0) else None) for v in a["cells"]] if isinstance(a.get("cells"), list) else None) if isinstance(a, dict) else None
            self.planes.append(q)
        self.side = max(4, rnd(self.ext / 100 * self.win_frac * self.N)); self.x0 = rnd((self.N - self.side) / 2)   # the frame's box: ext % of the window
        os_ = float(model.get("outScale", 1) or 1); self.out_scale = os_ if 0 < os_ < 1 else 1.0
        self.so = max(4, rnd(self.side * self.out_scale)) if self.out_scale < 1 else self.side; self.xo = rnd((self.N - self.so) / 2)   # the box the readout bins
        self._freq(); self._fixed(); self._trace = None; self._hids = []

    # ---- the wafers' roles
    @staticmethod
    def learned_any(q): return q["kind"] == "learned" or bool(q.get("kinds") and "learned" in q["kinds"])
    def role(self, q, sec=0):                                                  # (kind, param, arr) of a wafer in one section
        if not q.get("kinds"): return q["kind"], q["param"], q.get("arr")
        k = q["kinds"][sec] if sec < len(q["kinds"]) else "glass"
        prm = float((q["params"][sec] if q.get("params") and sec < len(q["params"]) else q["param"]) or 0)
        a = q["arrs"][sec] if q.get("arrs") and sec < len(q["arrs"]) else q.get("arr")
        return k, prm, (dict(nx=int(a.get("nx", 1) or 1), ny=int(a.get("ny", 1) or 1), pitch=float(a.get("pitch_mm", a.get("pitch", 0.5)) or 0.5), cells=a.get("cells")) if isinstance(a, dict) else None)
    def fixed_phase_for(self, i, sec):
        q = self.planes[i]
        if not q.get("kinds"): return self.fixed_phase[i]
        key = (i, sec)
        if key not in self._fixed_sec:
            kind, prm, arr = self.role(q, sec); self._fixed_sec[key] = None if kind in ("learned", "nl", "glass", "probe") else self.build_phase(kind, prm, arr)
        return self._fixed_sec[key]

    # ---- propagation
    def win_w(self): return self.N * self.dx * self.win_frac
    def _freq(self):
        N = self.Nm = 2 * self.N if self.mirror else self.N; dx = self.dx; df = 1 / (N * dx); inv = self.nmed / self.lam
        i = torch.arange(N); f = torch.where(i < N // 2, i, i - N).to(torch.float64) * df
        fx, fy = torch.meshgrid(f, f, indexing="xy"); f2 = fx * fx + fy * fy
        self.fmag = torch.sqrt(f2); self.inv = inv; self.df = df
        self.kz = torch.sqrt(torch.clamp(inv * inv - f2, min=0)); self.kzOK = (f2 < inv * inv)
    def H(self, z_um):                                                         # a hop of z µm in the medium, band-limited (Matsushima's limit)
        fLim = self.inv / math.sqrt((2 * self.df * abs(z_um)) ** 2 + 1)
        ok = self.kzOK & (self.fmag <= fLim)
        ph = TAU * z_um * self.kz
        H = torch.where(ok, torch.polar(torch.ones_like(ph), ph), torch.zeros_like(ph, dtype=torch.complex128))
        return H.to(torch.complex64).to(self.dev)
    def z_max(self):                                                           # the longest hop whose band limit is still the grid's Nyquist
        if getattr(self, "_zmax", None) is None:
            q = (2 * self.inv * self.dx) ** 2 - 1; self._zmax = (self.Nm * self.dx / 2) * math.sqrt(q) if q > 0 else float("inf")
        return self._zmax
    def H_cached(self, z_um):
        key = round(float(z_um), 3)
        if key not in self._H: self._H[key] = self.H(key)
        return self._H[key]
    def prop_z(self, E, z_um):                                                 # z µm in sub-hops (subHopUm / zMaxSplit), whole otherwise
        zc = min(self.z_max(), self.sub_hop) if self.sub_hop > 0 else (self.z_max() if self.zmax_split else float("inf"))
        n = max(1, int(math.ceil(z_um / zc))); H = self.H_cached(z_um / n)
        for _ in range(n): E = self.prop(E, H)
        return E
    def prop(self, E, H):                                                      # one hop: FFT · H · IFFT
        if not self.mirror:
            E = torch.fft.ifft2(torch.fft.fft2(E) * H)
            if self.absorb:
                if self._absorb_mask.device != E.device: self._absorb_mask = self._absorb_mask.to(E.device)
                E = E * self._absorb_mask
            return E
        Ex = torch.cat([E, -torch.flip(E, dims=(-2,))], dim=-2); Ex = torch.cat([Ex, -torch.flip(Ex, dims=(-1,))], dim=-1)   # reflecting sides: odd extension on a doubled grid
        return torch.fft.ifft2(torch.fft.fft2(Ex) * H)[..., :self.N, :self.N]
    def _fixed(self):
        self.multi = any(p.get("kinds") for p in self.planes)                  # per-section roles: unused wafers merge into the next hop
        self.Hs = [None] * len(self.planes) if self.multi else [self.H(p["dist"] * 1000) if p["dist"] > 0 else None for p in self.planes]
        self.fixed_phase = [None if (p["kind"] in ("learned", "nl", "glass", "probe") or p["off"]) else self.build_phase(p["kind"], p["param"], p.get("arr")) for p in self.planes]
        self._fixed_sec = {}; self._H = {}

    # ---- the surfaces
    def build_phase(self, kind, param, arr=None):                             # the fixed kinds
        N, dx = self.N, self.dx; c = (N - 1) / 2
        x = (torch.arange(N, dtype=torch.float64) - c) * dx; X, Y = torch.meshgrid(x, x, indexing="xy"); r2 = X * X + Y * Y
        if kind == "flat": v = torch.zeros_like(r2)
        elif kind == "lens": v = -math.pi * self.nmed * r2 / (self.lam * param * 1000)
        elif kind == "lensarr":                                                # nx × ny lenses of f = param mm, a pitch apart; each point takes its nearest lens
            a = arr or {}; nx = max(1, int(a.get("nx", 1) or 1)); ny = max(1, int(a.get("ny", 1) or 1)); P = max(1e-3, float(a.get("pitch", 0) or 0) or self.win_w() / 1000 / max(nx, ny)) * 1000
            ix = torch.clamp(torch.round(X / P + (nx - 1) / 2), 0, nx - 1); iy = torch.clamp(torch.round(Y / P + (ny - 1) / 2), 0, ny - 1); cx = (ix - (nx - 1) / 2) * P; cy = (iy - (ny - 1) / 2) * P
            u = X - cx; w = Y - cy; cells = a.get("cells")
            if isinstance(cells, list) and len(cells) == nx * ny:
                fmap = torch.tensor([float(c) if (c is not None and c > 0) else 0.0 for c in cells], dtype=torch.float64)[(iy * nx + ix).long()]
                v = torch.where(fmap > 0, -math.pi * self.nmed * (u * u + w * w) / (self.lam * torch.where(fmap > 0, fmap, torch.ones_like(fmap)) * 1000), torch.zeros_like(u))
            else: v = -math.pi * self.nmed * (u * u + w * w) / (self.lam * param * 1000)
        elif kind == "diverge": v = math.pi * self.nmed * r2 / (self.lam * param * 1000)
        elif kind == "grating": v = TAU * ((X / param) - torch.floor(X / param))
        elif kind == "axicon": v = -TAU * torch.sqrt(r2) / param
        elif kind == "vortex": v = rnd(param) * torch.atan2(Y, X)
        elif kind == "cylx": v = -math.pi * self.nmed * X * X / (self.lam * param * 1000)
        else: raise NotImplementedError("plane kind not supported: " + kind)
        return v.to(torch.float32).to(self.dev)
    def learned_phase(self, p, sec=0):                                        # each learned value fills its block of samples
        phi = (p.get("phis") or [None] * (sec + 1))[sec] if sec else p["phi"]
        N = self.N
        if phi is None: return torch.zeros(N, N, device=self.dev)
        n = int(phi.shape[0])
        if n == N: return phi.to(self.dev)
        q = N // n
        if q * n != N: raise ValueError(f"feature map {n}² does not tile the {N}² grid")
        return phi.to(self.dev).repeat_interleave(q, 0).repeat_interleave(q, 1)

    # ---- the polariton layer
    def nl_win(self): return min(1.0, self.nl_tau / max(self.nl_plen, 1e-15))
    def nl_pulse_energy(self):
        h_nu = 6.626e-34 * 2.998e8 / (self.lam * 1e-6); frac = max(1e-6, self.nl_frac if self.nl_frac else 1.0)
        return self.N * self.N * self.nl_se * h_nu / self.nl_qe / frac / max(self.nl_wshare, 1e-9) / self.nl_ppf
    def nl_gain(self, E, param, p_in=None):                                    # saturable complex transmission t(I) = √T·e^{iφ}
        I = E.real * E.real + E.imag * E.imag
        if self.nl_abs and p_in is not None:
            k = self.nl_fsat * 1e-9 * (self.dx * self.dx * 1e-8) / max(self.nl_pulse_energy() * max(self.nl_wshare, 1e-9) * self.nl_win(), 1e-300)
            Is = (p_in * k).clamp(min=1e-30)
        else:
            Is = I.mean(dim=(-2, -1), keepdim=True) * param / (self.win_frac ** 2)
        u = torch.pow(I / Is + 1, -self.nl["knee"]); mag = torch.sqrt(self.nl["tinf"] - u * (self.nl["tinf"] - self.nl["t0"])); ph = (1 - u) * self.nl["phi"]
        return torch.polar(mag, ph)

    # ---- one pass through one section
    @torch.no_grad()
    def forward(self, E, sec=0, pass_k=None):                                  # E [B,N,N] complex at the DMD → the field at the exit face
        tr = self._trace; k = sec if pass_k is None else pass_k
        p_in = (E.real * E.real + E.imag * E.imag).sum(dim=(-2, -1), keepdim=True) if self.nl_abs else None
        if self.in_gap > 0: E = self.prop_z(E, self.in_gap * 1000)
        pend = 0.0
        for i, p in enumerate(self.planes):
            kind, prm, _ = self.role(p, sec) if p.get("kinds") else (p["kind"], p["param"], None)
            if not p["off"] and kind != "glass":
                if pend > 0: E = self.prop_z(E, pend); pend = 0.0
                if tr is not None: tr.append({"pass": k, "plane": i, "kind": kind, "light": self.window_light(E)})   # the light arriving at this surface
                if kind == "nl": E = E * self.nl_gain(E, prm, p_in)
                elif kind != "probe":
                    phi = self.learned_phase(p, sec) if kind == "learned" else self.fixed_phase_for(i, sec)
                    if phi is not None: E = E * torch.polar(torch.ones_like(phi), phi)
            if self.multi: pend += p["dist"] * 1000
            elif self.Hs[i] is not None: E = self.prop_z(E, p["dist"] * 1000)
        if pend > 0: E = self.prop_z(E, pend)
        if tr is not None: tr.append({"pass": k, "plane": None, "kind": "exit", "light": self.window_light(E)})
        if self.nl_abs and p_in is not None:
            p_out = (E.real * E.real + E.imag * E.imag).sum(dim=(-2, -1), keepdim=True)
            self.nl_frac = float((p_out.sum() / p_in.sum().clamp(min=1e-30)).clamp(min=1e-6))
        return E

    # ---- the DMD and the camera
    def dmd(self, frames, hidden=False):                                       # frames [B,F,F] in 0..1 → the launched field [B,N,N]
        B, F, _ = frames.shape; N, side, x0 = self.N, self.side, self.x0
        E = torch.zeros(B, N, N, device=self.dev)
        ys = torch.arange(side, dtype=torch.float64); fi = torch.floor((ys + 0.5) * F / side).long().clamp(max=F - 1).to(self.dev)
        blk = frames.to(self.dev)[:, fi][:, :, fi]
        E[:, x0:x0 + side, x0:x0 + side] = blk
        if self.img["inv"] and not hidden: E[:, x0:x0 + side, x0:x0 + side] = 1 - blk
        if self.img["flip"]: E = torch.flip(E, dims=(1, 2))
        E = E.to(torch.complex64)
        if self.dmd_na > 0: E = torch.fft.ifft2(torch.fft.fft2(E) * self.dmd_pupil())
        return E
    def dmd_pupil(self):
        key = ("pupil", self.N, self.dx, self.dmd_na)
        if getattr(self, "_pupil_key", None) != key:
            f = torch.fft.fftfreq(self.N, d=self.dx); fx, fy = torch.meshgrid(f, f, indexing="ij")
            self._pupil = ((fx * fx + fy * fy) <= (self.dmd_na / self.lam) ** 2).to(torch.complex64).to(self.dev); self._pupil_key = key
        return self._pupil
    def nn_mat(self, side, F):                                                 # sample y of a box → frame pixel floor((y + ½)·F/side)
        key = ("nn", side, F)
        if getattr(self, "_nn_key", None) != key:
            R = torch.zeros(side, F); ys = torch.arange(side, dtype=torch.float64); fi = torch.floor((ys + 0.5) * F / side).long().clamp(max=F - 1); R[torch.arange(side), fi] = 1
            self._nn = R.to(self.dev); self._nn_key = key
        return self._nn
    def hidden(self, E, F, k=0):
        """The quad crystal's hand-off after pass k: the camera bins the exit box to F×F, the imager sets exposure and γ, and the
        picture is dithered to the DMD's levels (and the residual, the band and the un-rotation applied) for pass k+1. [B,F,F] in 0..1."""
        I = E.real * E.real + E.imag * E.imag; so, xo = self.so, self.xo
        R = self.nn_mat(so, F); v = torch.einsum("sf,bst,tg->bfg", R, I[:, xo:xo + so, xo:xo + so], R)
        full = self.side * self.side / (F * F)
        if self.q_norm:                                                        # auto-exposure: black at t·μ, white at μ + s·σ
            ell = v / full; mu = ell.mean(dim=(1, 2), keepdim=True)
            sd = (ell.var(dim=(1, 2), keepdim=True, unbiased=False) + 1e-18).sqrt()
            shift = self.q_norm["t"] * mu; a = (ell - shift) / (mu + self.q_norm["s"] * sd - shift).clamp_min(1e-9) - self.img["b"]
        else: a = (v / full) * self.q_gains[k] - self.img["b"]
        r = torch.pow(torch.clamp(a, 0, 1), float(self.img["gm"]))
        L = float(self.q_levels) - 1.0
        out = torch.round(r * L) / L
        if self.q_unrot: out = torch.flip(out, dims=(1, 2))
        if self.q_res:
            out = torch.round(torch.clamp(out + self.q_res * self._in.to(out.dtype), 0.0, 1.0) * L) / L
        if self.q_band:
            y0 = self.q_band_y0; out = torch.cat([out[:, :y0], self._board[:, y0:].to(out.dtype) * (self.q_band / L)], dim=1)
        return out
    @torch.no_grad()
    def run(self, frames):
        """Every pass: the frame in, the last pass's exit field out, with that pass's launched field. Read it with readout() + imager().
        frames: one picture [F, F] or a batch [B, F, F] in 0..1, as a numpy array or a torch tensor."""
        frames = torch.as_tensor(np.asarray(frames, dtype=np.float32) if not torch.is_tensor(frames) else frames, dtype=torch.float32)
        if frames.dim() == 2: frames = frames[None]
        frames = frames.to(self.dev); F = frames.shape[-1]; Fh = self.q_hidden or F; self._board = frames
        if Fh != F:
            assert F % Fh == 0, f"the input frame ({F}) must be a whole multiple of quad.hidden ({Fh})"
            self._in = torch.nn.functional.avg_pool2d(frames.unsqueeze(1), F // Fh).squeeze(1)
        else: self._in = frames
        E0 = self.dmd(frames)
        E = self.forward(E0, self.q_pass[0] if self.q_pass else 0, 0); self._hids = []
        for k in range(1, self.hops):
            q = self.hidden(E, Fh, k - 1); self._hids.append(q); self._in = q
            E0 = self.dmd(q, hidden=True)
            E = self.forward(E0, self.q_pass[k] if self.q_pass else k, k)
        return E, E0
    @property
    def hidden_pictures(self):
        """After run(): the camera's picture after each pass but the last, as written to the DMD for the next pass, [B,F,F] in 0..1 each."""
        return list(self._hids)
    def window_light(self, E, n=256):
        """|E|² over the lit window, block-averaged to about n × n: [B, n, n] on the CPU."""
        I = (E.real * E.real + E.imag * E.imag).detach(); w = max(1, rnd(self.N * self.win_frac)); x0 = (self.N - w) // 2
        q = max(1, w // n); return torch.nn.functional.avg_pool2d(I[:, x0:x0 + w, x0:x0 + w].float().unsqueeze(1), q).squeeze(1).cpu()
    def trace(self, frames, n=256):
        """run() with a record of the light inside the crystal: the exit field, plus [{"pass", "plane", "kind", "light"}] with the light
        arriving at every surface of every pass and at each pass's exit face (plane None, kind "exit"). Reading the light changes nothing."""
        self._trace = []
        try: E, _ = self.run(frames); return E, self._trace
        finally: self._trace = None
    def readout(self, E, F, E_in=None):                                       # the output box binned to the F×F sensor picture (sums of intensity) [B,F,F]
        I = E.real * E.real + E.imag * E.imag; side, x0 = self.so, self.xo
        R = self.readout_map(side, F)
        return torch.einsum("sf,bst,tg->bfg", R, I[:, x0:x0 + side, x0:x0 + side], R)
    @property
    def sensor_turned(self):
        """True when an output relay images the exit face onto the camera turned 180° (the camera sees the exit face upside down)."""
        sen = self.img.get("sen", "ideal")
        return (isinstance(sen, (int, float)) and not isinstance(sen, bool)) or sen in ("mirror1", "mirror2")
    def readout_map(self, side, F):                                            # sample columns → sensor columns, with weights
        sen = self.img.get("sen", "ideal"); P = 0.0
        if sen == "bonded": P = self.sens_pitch / self.dx
        elif isinstance(sen, str) and sen.startswith("mirror") and self.mirrors > 0:
            Pm = self.N / ((2 if sen == "mirror2" else 1) * self.mirrors); P = Pm if Pm > 1 else 0.0
        turned = (isinstance(sen, (int, float)) and not isinstance(sen, bool)) or sen in ("mirror1", "mirror2")   # an output relay turns the picture 180°
        fx = (lambda w: F - 1 - w) if turned else (lambda w: w)
        R = torch.zeros(side, F); fw = side / F
        for x in range(side):
            if P <= 0:
                w0 = min(F - 1, math.floor(x / fw)); w1 = min(F - 1, math.floor((x + 1 - 1e-9) / fw))
                if w1 == w0: R[x, fx(w0)] += 1
                else: cut = (w0 + 1) * fw; R[x, fx(w0)] += cut - x; R[x, fx(w1)] += x + 1 - cut
                continue
            s = math.floor(x / P); a = s * P; b = min(side, (s + 1) * P); w = max(0, math.floor(a / fw)); hit = False
            while w < F and w * fw < b:
                o = min(b, (w + 1) * fw) - max(a, w * fw)
                if o > 1e-9: R[x, fx(w)] += o / (b - a); hit = True
                w += 1
            if not hit: R[x, fx(min(F - 1, math.floor(x * F / side)))] += 1
        return R.to(self.dev)
    def imager(self, v):                                                       # r = clip(g·ℓ − black, 0, 1)^γ, ℓ = a pixel's light over a fully-on pixel's
        if self.img["mode"] == "off": return v
        F = v.shape[-1]; full = self.side * self.side / (F * F)
        return torch.pow(torch.clamp((v / full) * self.img["g"] - self.img["b"], 0, 1), float(self.img["gm"]))


def load(path, **kw):
    """A Crystal from a .safetensors crystal file: fp32, or packed at 16 or 8 bits."""
    return Crystal(read_model(path), **kw)
