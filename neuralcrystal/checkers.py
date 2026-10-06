"""The checkers crystal's problem: English draughts, the 96 × 96 picture the crystal is shown, and the 32 × 32 move map it answers on.

A position is drawn from the side to move's point of view (its men solid discs at the top, moving down; the opponent's as rings; a
king carries a crown), so every frame asks the same question: "these are mine, which of my moves is best?". Moves are ATOMIC: one
step or one jump. Captures are compulsory; a jump chain goes on one jump at a time with the same piece (Position.jump), and a man
crowned on the far row ends the turn there. The answer is light on a 32 × 32 grid of detector cells on the last section's exit face:
the row band is the from-square, the column band the to-square, both in the canonical (side-to-move-at-top) numbering.

    import neuralcrystal as nc
    from neuralcrystal import checkers as ck
    c = nc.load("checkers-crystal.safetensors", device="cpu")
    pos = ck.parse("bb.bb...b.wb.b.b....w.ww.w.ww.ww w - 0")
    move, scores = ck.best_move(c, pos)          # the brightest legal move, and every legal move's light
    print(ck.uci(move)); pos = ck.make(pos, move)

This is a port of the reference board code the crystals were trained with at its defaults: the 12-unit layout on the 96 frame, the "box2"
jump mark and the "fromto32" move map — the frame, rules and cells every published checkers crystal was trained and scored on.
"""
import math
import random
from typing import NamedTuple

import numpy as np
import torch

# ---- the board: squares 0 … 31 in reading order over the dark squares, row 0 at the top (draughts' 1–32, zero-based)
# sq[i]: 0 empty, +1 / +2 black man / king (starts on 0–11, moves DOWN), −1 / −2 white (20–31, moves up). Black moves first.
def row_of(sq): return sq >> 2
def col_of(sq): return 2 * (sq & 3) + (1 - (row_of(sq) & 1))
def sq_at(r, c): return -1 if (r < 0 or r > 7 or c < 0 or c > 7 or ((r + c) & 1) == 0) else 4 * r + (c >> 1)
def flip(sq): return 31 - sq                                                   # the 180° turn: row r col c → row 7−r col 7−c
DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
IDLE_DRAW = 80                                                                 # 40 moves a side with no capture and no man move


class Position(NamedTuple):
    sq: tuple                                                                  # 32 ints
    turn: int = 1                                                              # +1 black, −1 white
    jump: int = -1                                                             # −1, or the square whose piece must go on jumping
    idle: int = 0                                                              # plies since the last capture or man move
    def __str__(self): return to_str(self)


class Move(NamedTuple):
    frm: int                                                                   # real squares, 0 … 31
    to: int
    cap: int = -1                                                              # the captured square, or −1 for a step
    promo: bool = False                                                        # a man reaching the far row (crowned; the turn ends)
    king: bool = False                                                         # the piece moving is a king


_CH = {"b": 1, "B": 2, "w": -1, "W": -2}
_CHR = {1: "b", 2: "B", -1: "w", -2: "W", 0: "."}


def start(): return Position(tuple([1] * 12 + [0] * 8 + [-1] * 12), 1, -1, 0)
def parse(s):
    """'<32 chars of . b B w W> <b|w> <jump square 1-32 or -> <idle>' → Position."""
    p = str(s).split(); b = p[0] if p else ""
    sq = tuple(_CH.get(b[i], 0) if i < len(b) else 0 for i in range(32))
    turn = -1 if (p[1] if len(p) > 1 else "b") == "w" else 1
    jump = int(p[2]) - 1 if len(p) > 2 and p[2] != "-" else -1
    try: idle = int(p[3]) if len(p) > 3 else 0
    except ValueError: idle = 0
    return Position(sq, turn, jump, idle)
def to_str(pos): return "".join(_CHR[v] for v in pos.sq) + f" {'b' if pos.turn > 0 else 'w'} {'-' if pos.jump < 0 else pos.jump + 1} {pos.idle}"
def uci(m): return f"{m.frm + 1}{'x' if m.cap >= 0 else '-'}{m.to + 1}"          # draughts notation: 11-15 a step, 11x18 a jump


# ---- the rules
def _mine(pos, p): return p != 0 and (p > 0) == (pos.turn > 0)
def _jumps_from(pos, frm, out):
    p = pos.sq[frm]
    if not _mine(pos, p): return out
    r, c, king, fwd = row_of(frm), col_of(frm), abs(p) == 2, 1 if pos.turn > 0 else -1
    for dr, dc in DIRS:
        if not king and dr != fwd: continue
        mid, to = sq_at(r + dr, c + dc), sq_at(r + 2 * dr, c + 2 * dc)
        if mid < 0 or to < 0: continue
        q = pos.sq[mid]
        if not q or (q > 0) == (pos.turn > 0) or pos.sq[to]: continue
        out.append(Move(frm, to, mid, not king and row_of(to) == (7 if pos.turn > 0 else 0), king))
    return out
def _steps_from(pos, frm, out):
    p = pos.sq[frm]
    if not _mine(pos, p): return out
    r, c, king, fwd = row_of(frm), col_of(frm), abs(p) == 2, 1 if pos.turn > 0 else -1
    for dr, dc in DIRS:
        if not king and dr != fwd: continue
        to = sq_at(r + dr, c + dc)
        if to < 0 or pos.sq[to]: continue
        out.append(Move(frm, to, -1, not king and row_of(to) == (7 if pos.turn > 0 else 0), king))
    return out
def moves(pos):
    """Every legal ATOMIC move: mid-chain only that piece's jumps; otherwise every jump if any exists (captures are compulsory), else every step."""
    if pos.jump >= 0: return _jumps_from(pos, pos.jump, [])
    js = []
    for i in range(32): _jumps_from(pos, i, js)
    if js: return js
    st = []
    for i in range(32): _steps_from(pos, i, st)
    return st
def make(pos, m):
    """The position after an atomic move. While a jump chain goes on the turn does NOT pass (jump = the landing square)."""
    sq = list(pos.sq); p = sq[m.frm]; sq[m.frm] = 0
    if m.cap >= 0: sq[m.cap] = 0
    crown = m.promo and abs(p) == 1
    sq[m.to] = (2 if p > 0 else -2) if crown else p
    idle = 0 if (m.cap >= 0 or abs(p) == 1) else pos.idle + 1
    n = Position(tuple(sq), pos.turn, -1, idle)
    if m.cap >= 0 and not crown and _jumps_from(n, m.to, []): return n._replace(jump=m.to)
    return n._replace(turn=-pos.turn)
def status(pos):
    """(over, winner, reason): no legal move loses; 40 idle moves a side is a draw."""
    if not moves(pos): return True, -pos.turn, "blocked" if any(_mine(pos, v) for v in pos.sq) else "no pieces"
    if pos.idle >= IDLE_DRAW: return True, 0, "40-move rule"
    return False, 0, ""
def from_uci(pos, s):
    """'11-15' / '11x18' → the legal Move, or None."""
    t = str(s).strip().replace("x", "-").split("-")
    if len(t) != 2 or not t[0].isdigit() or not t[1].isdigit(): return None
    f, to = int(t[0]) - 1, int(t[1]) - 1
    return next((m for m in moves(pos) if m.frm == f and m.to == to), None)


# ---- the canonical view: the side to move at the top, drawn solid. White's positions are turned 180° and recoloured (sq → 31 − sq).
def view(pos):
    """(sq, jump) as drawn: + = the side to move."""
    if pos.turn > 0: return tuple(pos.sq), pos.jump
    s = [0] * 32
    for i in range(32): s[flip(i)] = -pos.sq[i]
    return tuple(s), (-1 if pos.jump < 0 else flip(pos.jump))
def view_sq(pos, sq): return sq if pos.turn > 0 else flip(sq)                 # a real square ↔ its drawn square (the map is its own inverse)


# ---- the frame: 96 × 96, 0 / 1. Only the 32 playable squares are marked, each by a corner bracket (a top arm and a left stroke, one
# pixel thick, 11 long, in a 12-pixel cell); a piece is an 8 × 8 glyph inset 2 px. The square whose piece must go on jumping has its
# bracket closed into a 2-px box and its piece drawn at 7 px.
FRAME, TILE, ARM, INS, BOX = 96, 12, 11, 2, 2
GLYPH = {1:  ["..####..", ".######.", "########", "########", "########", "########", ".######.", "..####.."],   # man: a disc
         2:  ["..####..", ".######.", "##.##.##", "##....##", "##....##", "########", ".######.", "..####.."],   # king: the crown cut dark
         -1: ["..####..", ".#....#.", "#......#", "#......#", "#......#", "#......#", ".#....#.", "..####.."],   # foe: a ring
         -2: ["..####..", ".#....#.", "#.#..#.#", "#.####.#", "#.####.#", "#......#", ".#....#.", "..####.."]}   # foe king: the crown lit
GLYPH7 = {1:  ["..###..", ".#####.", "#######", "#######", "#######", ".#####.", "..###.."],
          2:  ["..###..", ".#####.", "##.#.##", "##...##", "##...##", ".#####.", "..###.."],
          -1: ["..###..", ".#...#.", "#.....#", "#.....#", "#.....#", ".#...#.", "..###.."],
          -2: ["..###..", ".#...#.", "#.#.#.#", "#.###.#", "#.....#", ".#...#.", "..###.."]}


def _px(v, x, y):
    if 0 <= x < FRAME and 0 <= y < FRAME: v[y, x] = 1.0
def frame(pos):
    """The 96 × 96 picture of a position, float32 0 / 1."""
    v = np.zeros((FRAME, FRAME), dtype=np.float32); sq, jump = view(pos)
    for s in range(32):
        x0, y0 = col_of(s) * TILE, row_of(s) * TILE
        for i in range(ARM): _px(v, x0 + i, y0); _px(v, x0, y0 + i)
        if jump == s:                                                          # the box: 1 px outside the cell at top / left, 2 inside at bottom / right
            out, back = BOX - 1, TILE - BOX
            for i in range(-out, TILE):
                for j in range(BOX):
                    _px(v, x0 + i, y0 - out + j); _px(v, x0 + i, y0 + back + j); _px(v, x0 - out + j, y0 + i); _px(v, x0 + back + j, y0 + i)
        art = (GLYPH7 if jump == s else GLYPH).get(sq[s])
        if art:
            for y, row in enumerate(art):
                for x, ch in enumerate(row):
                    if ch == "#": _px(v, x0 + INS + x, y0 + INS + y)
    return v


# ---- the move map: 32 × 32 = 1,024 cells, cell = from × 32 + to in CANONICAL squares (a step, a jump and a chain's later jump alike)
GRID, PITCH, HALF = 32, 32 / 1024, 8 / 1024                                   # cell pitch and active half-side, as fractions of the window
def cell_of(frm, to): return frm * 32 + to
def cell_move(cell): return cell // 32, cell % 32                             # (from, to), canonical
def view_moves(pos, ms=None):
    """[(move, cell)] for the legal moves (or the ones given), the cell in canonical squares."""
    f = (lambda s: s) if pos.turn > 0 else flip
    return [(m, cell_of(f(m.frm), f(m.to))) for m in (moves(pos) if ms is None else ms)]
def legal_cells(pos):
    """{cell: move} over the legal moves, in move order."""
    out = {}
    for m, c in view_moves(pos): out[c] = m
    return out


def _r(x): return math.floor(x + 0.5)                                         # JavaScript's Math.round, as the simulator's detector rectangles
def bands(N, win_frac=1.0):
    """The cells' bands on the N × N grid: cell d covers samples [lo[d], lo[d] + side) on each axis. Rows = from, columns = to."""
    w = N * float(win_frac); c = (N - 1) / 2
    return [_r(c + ((d - (GRID - 1) / 2) * PITCH - HALF) * w) for d in range(GRID)], _r(2 * HALF * w)
def _band_matrix(crystal):
    key = (crystal.N, crystal.win_frac, str(crystal.dev))
    if getattr(crystal, "_ck_key", None) != key:
        lo, side = bands(crystal.N, crystal.win_frac); A = torch.zeros(crystal.N, GRID)
        for d, x0 in enumerate(lo):
            a, b = max(0, x0), min(crystal.N, x0 + side)
            if b > a: A[a:b, d] = 1.0
        crystal._ck_A = A.to(crystal.dev); crystal._ck_key = key
    return crystal._ck_A
def move_powers(crystal, E):
    """The light in each of the 1,024 cells from the exit field E [B, N, N] (or [N, N]) → [B, 1024] (Aᵀ·|E|²·A)."""
    if E.dim() == 2: E = E.unsqueeze(0)
    I = (E.real * E.real + E.imag * E.imag).float(); A = _band_matrix(crystal)
    return torch.einsum("yf,byx,xt->bft", A, I, A).reshape(I.shape[0], GRID * GRID)


def move_map(crystal, E):
    """The exit face's move map as a picture [B, 32, 32]: row = from-square, column = to-square (as the side to move sees the board)."""
    return move_powers(crystal, E).reshape(-1, GRID, GRID)


def move_scores(crystal, E_exit, pos):
    """Every legal move with its cell's light, brightest first: [(move, light)]. E_exit: one position's exit field [N, N] or [1, N, N]."""
    p = move_powers(crystal, E_exit)[0].cpu().double().numpy()
    ent = sorted(((c, m) for c, m in legal_cells(pos).items()), key=lambda cm: (-max(0.0, p[cm[0]]), cm[0]))
    return [(m, float(p[c])) for c, m in ent]


def read_move(p, pos, temp=0.0, top_k=0, rng=None):
    """The move the light names. p: the 1,024 cells' light.
    temp = 0 (the default): the brightest legal cell, ties to the lower cell number. temp > 0: sample among the top_k brightest legal
    cells (all of them if top_k is 0) with odds ∝ (light / brightest)^(1 / temp) — the file's checkers.playT / topK, which the simulator's
    game uses for variety; the published hold-out scores use temp = 0."""
    p = np.asarray(p, dtype=np.float64); lc = legal_cells(pos)
    if not lc: return None
    cand = sorted(((max(0.0, p[c]), c) for c in lc), key=lambda wc: (-wc[0], wc[1]))
    if top_k and len(cand) > top_k: cand = cand[:top_k]
    hit = cand[0]
    if temp > 0 and len(cand) > 1:
        mx = cand[0][0]; q = [(w / mx) ** (1 / temp) if mx > 0 else 1.0 for w, _ in cand]; Z = sum(q)
        r = (rng or random).random(); acc = 0.0; hit = cand[-1]
        for k, wc in enumerate(cand):
            acc += q[k] / Z
            if r <= acc: hit = wc; break
    return lc[hit[1]]


def _check(crystal):
    ck = crystal.m.get("checkers") or {}
    if int(ck.get("frame") or FRAME) != FRAME: raise NotImplementedError(f"checkers frame {ck.get('frame')}: only the 96 frame is supported")
    if str(ck.get("moveLayout") or "fromto32") != "fromto32" or str(ck.get("det") or "move32") != "move32":
        raise NotImplementedError("only the fromto32 / move32 move map is supported")


def frames(positions):
    """A batch of positions → float32 [B, 96, 96], ready for crystal.run."""
    return torch.from_numpy(np.stack([frame(p) for p in positions]))


def best_moves(crystal, positions, temp=0.0, top_k=0, rng=None):
    """Each position through the crystal and its move read: [(move, [(move, light)] brightest first)]. One forward pass for the batch."""
    _check(crystal)
    E, _ = crystal.run(frames(positions))
    P = move_powers(crystal, E).cpu().double().numpy(); out = []
    for pos, p in zip(positions, P):
        m = read_move(p, pos, temp, top_k, rng)
        sc = [(mv, float(p[c])) for c, mv in sorted(legal_cells(pos).items(), key=lambda cm: (-max(0.0, p[cm[0]]), cm[0]))]
        out.append((m, sc))
    return out


def best_move(crystal, pos, temp=0.0, top_k=0, rng=None):
    """(move, scores): the crystal's move on one position (None if it has no legal move) and every legal move's light, brightest first."""
    if isinstance(pos, str): pos = parse(pos)
    return best_moves(crystal, [pos], temp, top_k, rng)[0]
