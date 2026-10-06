"""The optics: how light travels through a NeuralCrystal, simulated in PyTorch.

A NeuralCrystal is a block of fused silica glass with thin etched surfaces stacked inside it. Each surface is a "phase map": at every
point it delays the light by a trained amount. Light that has been delayed differently at neighboring points interferes as it travels
on, so a stack of these surfaces can steer light from a picture going in to a pattern of bright spots coming out. Those spots are the
answer.

What this module simulates, step by step:

  1. The input. A micromirror array (a DMD: a chip of tiny tilting mirrors) shows a picture. Each lit mirror sends laser light into the
     glass, so the picture becomes a "light field": a grid of complex numbers whose size is the light's brightness and whose angle is
     its phase.
  2. Travel through glass. Between surfaces the light spreads out and interferes. We compute that with the angular-spectrum method:
     a Fourier transform splits the field into plane waves going in different directions, each direction picks up the phase it gains
     over the distance, and an inverse Fourier transform puts the field back together.
  3. The surfaces. At each etched surface the field is multiplied by e^(i·phase), using that surface's trained phase map. These phase
     maps are the "weights" stored in the crystal file.
  4. Several passes (CIFAR-10, checkers and chess). These crystals have four sections of glass. After each section a camera reads the
     light, its picture is adjusted for exposure and rounded to the brightness levels the mirrors can show, and the mirrors send that
     picture into the next section. This camera step is the only electronics in the loop, and it is what lets the crystal compute more
     than a single linear transform.
  5. The output. After the last section we return the light field leaving the glass. The problem modules (mnist.py, cifar10.py,
     checkers.py, chess.py) read their answers from how bright it is in different places.

The physics is scalar (one polarization), coherent (laser light) and at one wavelength (617 nm, red-orange). This is the same forward
pass the crystals were trained with: on the same input it gives the training code's output bit for bit. Options that no published
crystal uses are refused with a clear error rather than approximated.

Units: lengths in micrometers (µm) unless a name says mm. Arrays are shaped [batch, rows, columns].
"""
import math
import numpy as np
import torch

from .files import read_model

TAU = 2 * math.pi


def round_half_up(x):
    """Round like JavaScript's Math.round (halves go up). The crystals' geometry was drawn by JavaScript code, so the boxes below
    have to be rounded the same way or they land one sample off."""
    return int(math.floor(x + 0.5))


def pick_device(name=None):
    """The fastest place to run: an NVIDIA GPU, then an Apple-silicon GPU, then the CPU. Pass "cpu" (etc.) to choose."""
    if name: return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


class Crystal:
    """One crystal, ready to run. Build it with load("crystal.safetensors"), then call run(pictures)."""

    def __init__(self, model, device=None):
        self.m = model                                                             # the crystal file's description (geometry, camera, readout)
        self.dev = pick_device(device)
        self._refuse_unsupported(model)

        # ---- the simulation grid
        # The glass is simulated on a square grid of grid_size × grid_size points, sample_um apart. Only a central square "window"
        # of it is lit; the rest is the absorbing edge of the block.
        self.grid_size = int(model["N"])
        self.sample_um = float(model["dx_um"])
        self.wavelength_um = float(model.get("lam_um", 0.617))                       # in vacuum
        self.refractive_index = float(model.get("n_medium", 1.4607))                 # fused silica at 617 nm
        whole_block = bool(model.get("wholeGlass", False))
        window_um = float(model.get("winUm") or self.grid_size * self.sample_um)
        self.window_fraction = min(1.0, window_um / (self.grid_size * self.sample_um)) if whole_block else 1.0
        # The block's sides are blackened: light that wanders out of the window is absorbed after every step of travel.
        self.absorbing_edge = bool(model.get("absorbRim", False)) and whole_block and self.window_fraction < 1.0
        if self.absorbing_edge:
            width = max(1, round_half_up(self.grid_size * self.window_fraction)); start = round_half_up((self.grid_size - width) / 2)
            self._window_mask = torch.zeros(self.grid_size, self.grid_size)
            self._window_mask[start:start + width, start:start + width] = 1.0
        # Long gaps can be simulated in shorter steps, so that light near the edge can't wrap around the grid within one step.
        self.max_step_um = float(model.get("subHopUm") or 0.0)
        self.split_long_steps = bool(model.get("zMaxSplit", False))

        # ---- where the picture goes in and where the answer is read
        # The mirror picture fills a centered square box covering `ext` percent of the window. The camera reads the same box.
        input_percent = float(model.get("ext", 50))
        self.box_size = max(4, round_half_up(input_percent / 100 * self.window_fraction * self.grid_size))
        self.box_start = round_half_up((self.grid_size - self.box_size) / 2)

        # ---- the camera between passes and at the end
        camera = model.get("imager") or {}
        self.camera_gain = camera.get("g", 1)                                       # brightness multiplier (when not auto-exposed)
        self.camera_black = camera.get("b", 0.05)                                   # black level: light below this reads as zero
        self.camera_gamma = camera.get("gm", 1)                                     # response curve: reading = light ^ gamma
        self.camera_mode = camera.get("mode", "fixed")
        self.mirror_picture_flipped = camera.get("flip", True) is not False         # the mirrors show the picture turned 180°
        self.picture_inverted = bool(camera.get("inv", False))                      # input shown as a negative (1 − picture)
        self.sensor = camera.get("sen", "ideal")                                    # how the camera's pixels line up with the grid

        # ---- the passes (one for MNIST; four for CIFAR-10, checkers and chess)
        passes = model.get("quad") if isinstance(model.get("quad"), dict) else None
        self.passes = max(1, int(passes.get("n", 4) or 4)) if passes else 1
        mirrors_per_pixel = max(1, int(passes.get("mir", 4) or 4)) if passes else 4
        # A picture pixel shown by m × m mirrors can light 0 … m² of them: m² + 1 brightness levels.
        self.brightness_levels = max(2, int((passes.get("levels") if passes else 0) or (mirrors_per_pixel ** 2 + 1)))
        gains = [float(v) for v in (passes.get("gains") or [])] if passes else []
        self.pass_gains = (gains + [1.0] * self.passes)[:max(0, self.passes - 1)]     # fixed camera gain after each pass (when not auto-exposed)
        exposure = passes.get("norm") if passes else None
        # Auto-exposure: the camera sets black at t × the picture's mean brightness and white at mean + s × its spread.
        self.auto_exposure = {"t": float(exposure.get("t", 0) or 0), "s": float(exposure.get("s", 4) or 4)} if isinstance(exposure, dict) else None
        # Residual: a fraction of each section's own input picture is added back onto the picture it produces.
        self.residual = float(passes.get("res", 0) or 0) if passes else 0.0
        # Some crystals turn the camera's picture back the right way up before showing it to the next section.
        self.unflip_between_passes = bool(passes.get("unrot", False)) if passes else False

        # ---- the etched surfaces, in order from the mirrors to the camera
        self.input_gap_um = float(model.get("inGap_mm", 0)) * 1000                  # glass between the mirrors' image and the first surface
        self.surfaces = []
        for plane in model["planes"]:
            phase_maps = [plane.get("phi")] + list((plane.get("phis") or [])[1:])   # one trained phase map per section of glass
            self.surfaces.append({
                "phase_maps": [None if p is None else torch.tensor(np.array(p, dtype=np.float32)) for p in phase_maps],
                "gap_after_um": float(plane.get("dist_mm", 0) or 0) * 1000,          # glass between this surface and the next
                "off": bool(plane.get("off", False)),
            })

        self._prepare_propagation()
        self._step_cache = {}                                                        # transfer functions, by step length
        self._trace = None                                                           # set by trace(): a list to record the light into
        self._camera_pictures = []                                                   # the pictures between passes, from the last run

    @staticmethod
    def _refuse_unsupported(model):
        """Stop early, with a clear message, on options the published crystals don't use and this module doesn't simulate."""
        passes = model.get("quad") if isinstance(model.get("quad"), dict) else {}
        problems = []
        if isinstance(model.get("overlay"), dict) and model["overlay"].get("on"): problems.append("the overlay light path")
        if passes.get("bnd"): problems.append("attention boundaries between passes")
        if float(passes.get("ghost", 0) or 0): problems.append("adding the input board back at every pass")
        if float(passes.get("band", 0) or 0): problems.append("a fixed residual band between passes")
        if passes.get("passes"): problems.append("reusing sections of glass")
        if int(passes.get("hidden", 0) or 0): problems.append("camera pictures at a different size from the input")
        if model.get("solveMirror") and not model.get("sideBlack"): problems.append("mirrored block sides")
        if float(model.get("dmdNA", 0) or 0): problems.append("a limited-aperture mirror relay")
        if 0 < float(model.get("outScale", 1) or 1) < 1: problems.append("a reduced readout box")
        for plane in model["planes"]:
            if plane.get("kinds"): problems.append("different surface types in different sections")
            elif plane["kind"] != "learned" and not plane.get("off"): problems.append(f"a fixed '{plane['kind']}' surface")
        sensor = (model.get("imager") or {}).get("sen", "ideal")
        if sensor == "bonded" or (isinstance(sensor, str) and sensor.startswith("mirror")): problems.append(f"the '{sensor}' camera layout")
        if problems: raise NotImplementedError("this crystal uses features the neuralcrystal package doesn't simulate: " + "; ".join(sorted(set(problems))))

    # Names the problem modules (mnist.py, chess.py, …) read from a crystal.
    @property
    def N(self): return self.grid_size
    @property
    def win_frac(self): return self.window_fraction
    @property
    def hops(self): return self.passes
    @property
    def q_levels(self): return self.brightness_levels

    # ================================================================================================================ travel through glass

    def _prepare_propagation(self):
        """Everything that doesn't depend on the distance traveled: the spatial frequencies of the grid.

        A Fourier transform writes the light field as a sum of plane waves. The plane wave at spatial frequency (fx, fy) — cycles per µm
        across the grid — travels at an angle, and along the block it oscillates with frequency kz = √((n/λ)² − fx² − fy²). Waves with
        fx² + fy² > (n/λ)² don't travel at all ("evanescent") and are dropped."""
        size = self.grid_size
        self.freq_step = 1 / (size * self.sample_um)                                # cycles per µm between neighboring frequencies
        self.freq_in_glass = self.refractive_index / self.wavelength_um              # n / λ: the light's spatial frequency in the glass
        index = torch.arange(size)
        freqs = torch.where(index < size // 2, index, index - size).to(torch.float64) * self.freq_step   # FFT order: 0, +, …, −
        fx, fy = torch.meshgrid(freqs, freqs, indexing="xy")
        transverse_sq = fx * fx + fy * fy
        self.transverse_freq = torch.sqrt(transverse_sq)
        self.axial_freq = torch.sqrt(torch.clamp(self.freq_in_glass ** 2 - transverse_sq, min=0))   # kz, in cycles per µm
        self.travels = transverse_sq < self.freq_in_glass ** 2                     # False for evanescent waves

    def transfer_function(self, distance_um):
        """What traveling distance_um through the glass does to each plane wave: a phase delay of 2π·kz·distance.

        "Band-limited" (Matsushima & Shimobaba, 2009): on a finite grid, plane waves steeper than a limit that depends on the distance
        would alias, so they are cut off too."""
        limit = self.freq_in_glass / math.sqrt((2 * self.freq_step * abs(distance_um)) ** 2 + 1)
        keep = self.travels & (self.transverse_freq <= limit)
        delay = TAU * distance_um * self.axial_freq
        H = torch.where(keep, torch.polar(torch.ones_like(delay), delay), torch.zeros_like(delay, dtype=torch.complex128))
        return H.to(torch.complex64).to(self.dev)

    def _cached_transfer_function(self, distance_um):
        key = round(float(distance_um), 3)
        if key not in self._step_cache: self._step_cache[key] = self.transfer_function(key)
        return self._step_cache[key]

    def _longest_clean_step_um(self):
        """The longest step whose band limit is still the grid's own resolution; longer steps lose steep light."""
        q = (2 * self.freq_in_glass * self.sample_um) ** 2 - 1
        return (self.grid_size * self.sample_um / 2) * math.sqrt(q) if q > 0 else float("inf")

    def travel(self, field, distance_um):
        """Carry the light field distance_um through the glass, in one step or in several equal shorter ones."""
        if self.max_step_um > 0: step_limit = min(self._longest_clean_step_um(), self.max_step_um)
        elif self.split_long_steps: step_limit = self._longest_clean_step_um()
        else: step_limit = float("inf")
        steps = max(1, int(math.ceil(distance_um / step_limit)))
        H = self._cached_transfer_function(distance_um / steps)
        for _ in range(steps): field = self._one_step(field, H)
        return field

    def _one_step(self, field, H):
        """One step of travel: to plane waves (FFT), delay each one (× H), back to a field (inverse FFT), then lose whatever reached the
        blackened sides of the block."""
        field = torch.fft.ifft2(torch.fft.fft2(field) * H)
        if self.absorbing_edge:
            if self._window_mask.device != field.device: self._window_mask = self._window_mask.to(field.device)
            field = field * self._window_mask
        return field

    # ================================================================================================================ the etched surfaces

    def phase_map(self, surface, section):
        """A surface's trained phase map (radians) for one section of glass, on the full grid. A map stored coarser than the grid
        (e.g. 512 values across a 1024 grid) covers it in blocks: each stored value applies to a 2 × 2 patch of grid points."""
        maps = surface["phase_maps"]
        phase = maps[section] if section < len(maps) else None
        if phase is None: return torch.zeros(self.grid_size, self.grid_size, device=self.dev)
        stored = int(phase.shape[0])
        if stored == self.grid_size: return phase.to(self.dev)
        block = self.grid_size // stored
        if block * stored != self.grid_size: raise ValueError(f"a {stored} × {stored} phase map does not tile the {self.grid_size} × {self.grid_size} grid")
        return phase.to(self.dev).repeat_interleave(block, 0).repeat_interleave(block, 1)

    # ================================================================================================================ one pass through one section

    @torch.no_grad()
    def through_section(self, field, section=0):
        """Carry a light field [B, N, N] from the mirrors through one section of glass: the input gap, then for each etched surface,
        delay the light by its phase map and travel on to the next. Returns the light leaving the section."""
        if self.input_gap_um > 0: field = self.travel(field, self.input_gap_um)
        for number, surface in enumerate(self.surfaces):
            if not surface["off"]:
                if self._trace is not None:                                         # trace(): record the light arriving at this surface
                    self._trace.append({"pass": section, "plane": number, "kind": "learned", "light": self.window_light(field)})
                phase = self.phase_map(surface, section)
                field = field * torch.polar(torch.ones_like(phase), phase)          # × e^(i·phase): the etched delay
            if surface["gap_after_um"] > 0: field = self.travel(field, surface["gap_after_um"])
        if self._trace is not None:
            self._trace.append({"pass": section, "plane": None, "kind": "exit", "light": self.window_light(field)})
        return field

    # ================================================================================================================ the mirrors

    def mirrors(self, pictures, between_passes=False):
        """The light field the micromirror array launches for pictures [B, F, F] (brightness 0 … 1).

        Each picture pixel covers a block of grid points inside the input box (nearest-neighbor scaling, with half-pixel centers),
        and its brightness becomes the light's amplitude there. The relay optics show the picture turned 180°."""
        batch, F, _ = pictures.shape
        field = torch.zeros(batch, self.grid_size, self.grid_size, device=self.dev)
        rows = torch.arange(self.box_size, dtype=torch.float64)
        pixel = torch.floor((rows + 0.5) * F / self.box_size).long().clamp(max=F - 1).to(self.dev)   # which picture pixel each grid row shows
        box = pictures.to(self.dev)[:, pixel][:, :, pixel]
        start, end = self.box_start, self.box_start + self.box_size
        field[:, start:end, start:end] = box
        if self.picture_inverted and not between_passes: field[:, start:end, start:end] = 1 - box
        if self.mirror_picture_flipped: field = torch.flip(field, dims=(1, 2))
        return field.to(torch.complex64)

    # ================================================================================================================ the camera

    def _bin_to_pixels(self, box_size, F):
        """A [box_size, F] 0/1 matrix sending each grid row (or column) of the box to the camera pixel it falls in."""
        key = (box_size, F)
        if getattr(self, "_bin_key", None) != key:
            rows = torch.arange(box_size, dtype=torch.float64)
            pixel = torch.floor((rows + 0.5) * F / box_size).long().clamp(max=F - 1)
            M = torch.zeros(box_size, F); M[torch.arange(box_size), pixel] = 1
            self._bin, self._bin_key = M.to(self.dev), key
        return self._bin

    def camera_between_passes(self, field, F, after_pass=0):
        """What the camera hands to the mirrors for the next pass: a picture [B, F, F] with brightness 0 … 1.

        1. Brightness: |field|² summed over each camera pixel's patch of the box.
        2. Exposure: with auto-exposure, black is set at t × the picture's mean and white at the mean plus s × its spread; otherwise a
           fixed gain. Then the black level is subtracted.
        3. Response: clipped to 0 … 1 and raised to the camera's gamma.
        4. The mirrors can only show a few brightness levels (each pixel is a little grid of on/off mirrors), so round to those.
        5. Optionally turn the picture the right way up, and add back a fraction of this section's own input (the residual)."""
        brightness = field.real * field.real + field.imag * field.imag
        start, end = self.box_start, self.box_start + self.box_size
        M = self._bin_to_pixels(self.box_size, F)
        picture = torch.einsum("sf,bst,tg->bfg", M, brightness[:, start:end, start:end], M)
        full_pixel = self.box_size * self.box_size / (F * F)                       # the light a fully lit pixel would collect
        if self.auto_exposure:
            level = picture / full_pixel
            mean = level.mean(dim=(1, 2), keepdim=True)
            spread = (level.var(dim=(1, 2), keepdim=True, unbiased=False) + 1e-18).sqrt()
            black = self.auto_exposure["t"] * mean
            exposed = (level - black) / (mean + self.auto_exposure["s"] * spread - black).clamp_min(1e-9) - self.camera_black
        else:
            exposed = (picture / full_pixel) * self.pass_gains[after_pass] - self.camera_black
        response = torch.pow(torch.clamp(exposed, 0, 1), float(self.camera_gamma))
        top = float(self.brightness_levels) - 1.0
        shown = torch.round(response * top) / top                                   # the nearest brightness the mirrors can show
        if self.unflip_between_passes: shown = torch.flip(shown, dims=(1, 2))
        if self.residual:
            shown = torch.round(torch.clamp(shown + self.residual * self._section_input.to(shown.dtype), 0.0, 1.0) * top) / top
        return shown

    # ================================================================================================================ running the crystal

    @torch.no_grad()
    def run(self, pictures):
        """The whole simulation: the pictures go in on the mirrors, the light makes every pass through the glass (with the camera
        between passes), and the light field leaving the glass comes out.

        pictures: one picture [F, F] or a batch [B, F, F], brightness 0 … 1, as a numpy array or a torch tensor.
        Returns (light leaving the glass [B, N, N], the light the mirrors launched into the last pass [B, N, N]); complex."""
        if not torch.is_tensor(pictures): pictures = np.asarray(pictures, dtype=np.float32)
        pictures = torch.as_tensor(pictures, dtype=torch.float32)
        if pictures.dim() == 2: pictures = pictures[None]
        pictures = pictures.to(self.dev); F = pictures.shape[-1]
        self._section_input = pictures                                              # what the residual adds back
        launched = self.mirrors(pictures)
        field = self.through_section(launched, 0)
        self._camera_pictures = []
        for section in range(1, self.passes):
            picture = self.camera_between_passes(field, F, section - 1)
            self._camera_pictures.append(picture); self._section_input = picture
            launched = self.mirrors(picture, between_passes=True)
            field = self.through_section(launched, section)
        return field, launched

    @property
    def hidden_pictures(self):
        """After run(): the camera's picture after each pass but the last, as the mirrors showed it to the next pass ([B, F, F] each)."""
        return list(self._camera_pictures)

    def trace(self, pictures, n=256):
        """run(), also recording the light arriving at every surface of every pass and leaving each section. Returns (light leaving the
        glass, [{"pass", "plane", "kind", "light"}]); "light" is brightness over the window, about n × n. Recording changes nothing."""
        self._trace, self._trace_size = [], n
        try: field, _ = self.run(pictures); return field, self._trace
        finally: self._trace = None

    def window_light(self, field, n=None):
        """Brightness |field|² over the lit window, averaged down to about n × n pixels: [B, n, n] on the CPU."""
        n = n or getattr(self, "_trace_size", 256)
        brightness = (field.real * field.real + field.imag * field.imag).detach()
        width = max(1, round_half_up(self.grid_size * self.window_fraction)); start = (self.grid_size - width) // 2
        block = max(1, width // n)
        window = brightness[:, start:start + width, start:start + width].float().unsqueeze(1)
        return torch.nn.functional.avg_pool2d(window, block).squeeze(1).cpu()

    # ================================================================================================================ reading the answer as a picture

    @property
    def sensor_turned(self):
        """True when a relay lens images the exit face onto the camera turned 180° (MNIST's does)."""
        return isinstance(self.sensor, (int, float)) and not isinstance(self.sensor, bool)

    def readout(self, field, F):
        """The light leaving the glass as an F × F camera picture: brightness summed over each pixel's patch of the box [B, F, F]."""
        brightness = field.real * field.real + field.imag * field.imag
        start, end = self.box_start, self.box_start + self.box_size
        M = self._readout_matrix(self.box_size, F)
        return torch.einsum("sf,bst,tg->bfg", M, brightness[:, start:end, start:end], M)

    def _readout_matrix(self, box_size, F):
        """[box_size, F]: how much of each grid column falls in each camera column (a grid column straddling two pixels is shared by
        length). Turned around when the relay turns the picture."""
        M = torch.zeros(box_size, F); pixel_width = box_size / F
        pixel = (lambda w: F - 1 - w) if self.sensor_turned else (lambda w: w)
        for x in range(box_size):
            first = min(F - 1, math.floor(x / pixel_width)); last = min(F - 1, math.floor((x + 1 - 1e-9) / pixel_width))
            if first == last: M[x, pixel(first)] += 1
            else: cut = (first + 1) * pixel_width; M[x, pixel(first)] += cut - x; M[x, pixel(last)] += x + 1 - cut
        return M.to(self.dev)

    def imager(self, picture):
        """The camera's response to a readout picture: clip(gain × light − black, 0, 1) ^ gamma, light measured against a fully lit pixel."""
        if self.camera_mode == "off": return picture
        F = picture.shape[-1]; full_pixel = self.box_size * self.box_size / (F * F)
        return torch.pow(torch.clamp((picture / full_pixel) * self.camera_gain - self.camera_black, 0, 1), float(self.camera_gamma))


def load(path, **kw):
    """A Crystal from a .safetensors crystal file: 32-bit, or packed at 16 or 8 bits. device="cpu" (etc.) to choose where it runs."""
    return Crystal(read_model(path), **kw)
