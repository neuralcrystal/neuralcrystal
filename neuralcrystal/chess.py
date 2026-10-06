"""The chess crystal's problem: the rules, the picture the crystal is shown, the 64 × 64 move map it answers on, its win / draw / loss
readout, and the 2-ply "think" look-ahead built from those two answers.

The crystal plays ONE colour (the file's chess.playAs, black for every published chess crystal) and only ever sees that colour to move:
a position with the other side to move is MIRRORED first (ranks flipped a1 ↔ a8, colours swapped), so every frame asks the same
question. The frame is a compact 96-px board art — a flat board (no checkerboard), white pieces as filled 8-px sprites, black
pieces as their outlines, the side to move's clock lamp, the castling flags, an en-passant dot and the crystal's quartz mark beside its
own clock — with every pixel doubled to the 192 × 192 DMD frame ("art96x2"). The answer is light on a 64 × 64 grid of detector cells
on the last section's exit face: the row band is the from-square, the column band the to-square (a1 = 0 … h8 = 63, as the crystal sees
the board). A move never starts and ends on one square, so the diagonal is free: three runs of it are the value readout — the shares
of light on the win, draw and loss cells are the side to move's W / D / L.

    import neuralcrystal as nc
    from neuralcrystal import chess as ch
    c = nc.load("chess-crystal.safetensors", device="cpu")
    pos = ch.parse("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    move, scores, wdl, think = ch.best_move(c, pos)              # the brightest legal move, every legal move's light, the W/D/L
    move, scores, wdl, think = ch.best_move(c, pos, think=(3, 5)) # the 2-ply look-ahead: our top 3, the crystal's top 5 replies each
    print(ch.uci(move), ch.san(pos, move)); pos = ch.make(pos, move)

This is a port of the reference board code the crystals were trained with — its rules, its compact renderer and its move-map cells — and of the
trainer's play path: the frames, cells and look-ahead every published chess
number was measured with. Promotions share one cell and read as the queen.
"""
import math
import random
from typing import NamedTuple

import numpy as np
import torch

# ---------------------------------------------------------------------------------------------------------------- the position
# sq[i], i = a1 0 … h8 63: 0 empty, +1 … +6 white P N B R Q K, −1 … −6 black. castle: a subset of "KQkq" in that order.
PIECES = "PNBRQK"
PROMO_Q = 5


class Position(NamedTuple):
    sq: tuple                                                                  # 64 ints
    turn: str = "w"                                                            # "w" or "b"
    castle: str = "KQkq"                                                       # the rights held, "" for none
    ep: int = -1                                                               # the en-passant target square (as the FEN gives it), or −1
    half: int = 0                                                              # plies since the last capture or pawn move
    full: int = 1
    def __str__(self): return fen(self)


class Move(NamedTuple):
    frm: int                                                                   # squares 0 … 63
    to: int
    piece: int                                                                 # the piece moving (signed)
    capture: bool = False
    promo: int = 0                                                             # 2 … 5 (N B R Q), 0 for none
    castle: str = ""                                                           # "K" / "Q" for castling (the king's move), else ""
    ep: bool = False
    double: bool = False                                                       # a pawn's two-square push


class Status(NamedTuple):
    over: bool
    result: str                                                                # "1-0", "0-1", "1/2-1/2", or "" while the game goes on
    why: str                                                                   # checkmate / stalemate / 50-move rule / threefold repetition / insufficient material
    check: bool


def sq_name(s): return "abcdefgh"[s & 7] + str(1 + (s >> 3))
def sq_index(s): return (ord(s[0]) - 97) + 8 * (ord(s[1]) - 49)
def is_light(s): return (((s & 7) + (s >> 3)) & 1) == 1                       # a1 dark, h1 light
def piece_char(p): return PIECES[p - 1] if p > 0 else PIECES[-p - 1].lower() if p < 0 else "."


START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
def start(): return parse(START)
def parse(s):
    """A FEN → Position (castling rights and the en-passant square are taken as written)."""
    if isinstance(s, Position): return s
    f = str(s).strip().split(); rows = f[0].split("/") if f else []
    if len(rows) != 8: raise ValueError("bad FEN: " + str(s))
    sq = [0] * 64
    for r, row in enumerate(rows):
        file = 0
        for chr_ in row:
            if "1" <= chr_ <= "8": file += int(chr_); continue
            i = PIECES.find(chr_.upper())
            if i < 0: raise ValueError("bad FEN piece " + chr_)
            sq[(7 - r) * 8 + file] = i + 1 if chr_.isupper() else -(i + 1); file += 1
    cs = f[2] if len(f) > 2 else "-"
    castle = "".join(k for k in "KQkq" if k in cs)
    ep = sq_index(f[3]) if len(f) > 3 and f[3] != "-" else -1
    def num(i, d):
        try: return int(f[i]) if len(f) > i else d
        except ValueError: return d
    return Position(tuple(sq), "b" if len(f) > 1 and f[1] == "b" else "w", castle, ep, num(4, 0), num(5, 1))
def fen(pos):
    """Position → FEN (the en-passant square only when the capture is legal)."""
    out = []
    for r in range(7, -1, -1):
        e = 0; s = ""
        for f in range(8):
            p = pos.sq[r * 8 + f]
            if not p: e += 1; continue
            if e: s += str(e); e = 0
            s += piece_char(p)
        if e: s += str(e)
        out.append(s)
    ep = sq_name(pos.ep) if pos.ep >= 0 and any(m.ep for m in moves(pos)) else "-"
    return f"{'/'.join(out)} {pos.turn} {pos.castle or '-'} {ep} {pos.half} {pos.full}"
def key(pos): return " ".join(fen(pos).split()[:4])                           # the position's identity for repetition (no move counters)
def uci(m): return None if m is None else sq_name(m.frm) + sq_name(m.to) + (PIECES[m.promo - 1].lower() if m.promo else "")
def from_uci(pos, u):
    """'e2e4' / 'e7e8q' → the legal Move, or None."""
    pos = parse(pos); u = str(u).strip()
    return next((m for m in moves(pos) if uci(m) == u), None)


# ---------------------------------------------------------------------------------------------------------------- the rules
KN = ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1))
KG = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
BI = ((1, 1), (1, -1), (-1, 1), (-1, -1))
RO = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIRS = {2: KN, 3: BI, 4: RO, 5: KG, 6: KG}
def _inb(f, r): return 0 <= f < 8 and 0 <= r < 8
def _col(pos): return 1 if pos.turn == "w" else -1
def _sign(x): return (x > 0) - (x < 0)


def attacked(s, by, board):
    """Is square s attacked by colour `by` (+1 white, −1 black) on this board?"""
    f, r = s & 7, s >> 3
    pr = r - by
    if 0 <= pr < 8:
        for df in (-1, 1):
            pf = f + df
            if _inb(pf, pr) and board[pr * 8 + pf] == by: return True
    for df, dr in KN:
        F, R = f + df, r + dr
        if _inb(F, R) and board[R * 8 + F] == 2 * by: return True
    for df, dr in KG:
        F, R = f + df, r + dr
        if _inb(F, R) and board[R * 8 + F] == 6 * by: return True
    for dirs, a, b in ((BI, 3, 5), (RO, 4, 5)):
        for df, dr in dirs:
            F, R = f + df, r + dr
            while _inb(F, R):
                p = board[R * 8 + F]
                if p:
                    if p == a * by or p == b * by: return True
                    break
                F += df; R += dr
    return False
def king_sq(board, col):
    try: return board.index(6 * col)
    except ValueError: return -1
def in_check(pos):
    col = _col(pos); k = king_sq(pos.sq, col)
    return k >= 0 and attacked(k, -col, pos.sq)


def pseudo(pos):
    """The moves the pieces allow, before the 'do not leave your king in check' rule, in the reference order."""
    col = _col(pos); b = pos.sq; out = []
    def add(frm, to, **kw):
        kw.setdefault("capture", b[to] != 0); out.append(Move(frm, to, b[frm], **kw))
    for frm in range(64):
        p = b[frm]
        if not p or _sign(p) != col: continue
        f, r, kind = frm & 7, frm >> 3, abs(p)
        if kind == 1:
            r1, st, last = r + col, (1 if col > 0 else 6), (7 if col > 0 else 0)
            if _inb(f, r1) and not b[r1 * 8 + f]:
                to = r1 * 8 + f
                if (to >> 3) == last:
                    for pr in (5, 2, 4, 3): add(frm, to, promo=pr)
                else: add(frm, to)
                r2 = r + 2 * col
                if r == st and not b[r2 * 8 + f]: add(frm, r2 * 8 + f, double=True)
            for df in (-1, 1):
                F = f + df
                if not _inb(F, r1): continue
                to = r1 * 8 + F; q = b[to]
                if q and _sign(q) == -col:
                    if (to >> 3) == last:
                        for pr in (5, 2, 4, 3): add(frm, to, promo=pr, capture=True)
                    else: add(frm, to, capture=True)
                elif to == pos.ep and not q: add(frm, to, ep=True, capture=True)
            continue
        slide = 3 <= kind <= 5
        for df, dr in _DIRS[kind]:
            F, R = f + df, r + dr
            while _inb(F, R):
                to = R * 8 + F; q = b[to]
                if q:
                    if _sign(q) == -col: add(frm, to)
                    break
                add(frm, to)
                if not slide: break
                F += df; R += dr
        if kind == 6:                                                          # castling: the rights, the squares between empty, not in check, the king's path not attacked
            home = 4 if col > 0 else 60
            if frm != home: continue
            ks, qs = ("K" if col > 0 else "k") in pos.castle, ("Q" if col > 0 else "q") in pos.castle
            if ks and not b[home + 1] and not b[home + 2] and b[home + 3] == 4 * col and not attacked(home, -col, b) and not attacked(home + 1, -col, b) and not attacked(home + 2, -col, b):
                add(frm, home + 2, castle="K")
            if qs and not b[home - 1] and not b[home - 2] and not b[home - 3] and b[home - 4] == 4 * col and not attacked(home, -col, b) and not attacked(home - 1, -col, b) and not attacked(home - 2, -col, b):
                add(frm, home - 2, castle="Q")
    return out


def make(pos, m):
    """The position after a move. The en-passant square is set only when an enemy pawn stands beside the pawn."""
    pos = parse(pos)
    b = list(pos.sq); col = _col(pos); p = b[m.frm]; kind = abs(p)
    half = 0 if (kind == 1 or b[m.to]) else pos.half + 1; ep = -1
    b[m.to] = m.promo * col if m.promo else p; b[m.frm] = 0
    if m.ep: b[m.to - 8 * col] = 0
    if m.castle == "K": b[m.frm + 1] = b[m.frm + 3]; b[m.frm + 3] = 0
    if m.castle == "Q": b[m.frm - 1] = b[m.frm - 4]; b[m.frm - 4] = 0
    if m.double:
        f, r = m.to & 7, m.to >> 3
        for df in (-1, 1):
            if _inb(f + df, r) and b[r * 8 + f + df] == -col: ep = m.to - 8 * col
    cs = set(pos.castle)
    if kind == 6: cs -= {"K", "Q"} if col > 0 else {"k", "q"}
    def rook_home(s, c):
        if c > 0:
            if s == 7: cs.discard("K")
            if s == 0: cs.discard("Q")
        else:
            if s == 63: cs.discard("k")
            if s == 56: cs.discard("q")
    if kind == 4: rook_home(m.frm, col)
    rook_home(m.to, -col)                                                      # a rook captured on its home square loses that right
    return Position(tuple(b), "b" if col > 0 else "w", "".join(k for k in "KQkq" if k in cs), ep, half, pos.full + (1 if col < 0 else 0))


def moves(pos):
    """Every legal move, in the reference order (a promotion is four moves: Q, N, R, B)."""
    pos = parse(pos); col = _col(pos); out = []
    for m in pseudo(pos):
        n = make(pos, m); k = king_sq(n.sq, col)
        if k >= 0 and not attacked(k, -col, n.sq): out.append(m)
    return out


def status(pos, keys=None):
    """checkmate, stalemate, the 50-move rule (half ≥ 100), threefold repetition (`keys`: the game's position keys so
    far, key() of each, the current one included) and the reference's bare-material draw."""
    pos = parse(pos); ms = moves(pos); check = in_check(pos)
    if not ms: return Status(True, ("0-1" if pos.turn == "w" else "1-0") if check else "1/2-1/2", "checkmate" if check else "stalemate", check)
    if pos.half >= 100: return Status(True, "1/2-1/2", "50-move rule", check)
    if keys:
        k = key(pos)
        if sum(1 for q in keys if q == k) >= 3: return Status(True, "1/2-1/2", "threefold repetition", check)
    men = [(p, i) for i, p in enumerate(pos.sq) if p and abs(p) != 6]
    bare = (not men or (len(men) == 1 and abs(men[0][0]) in (2, 3))
            or (len(men) == 2 and abs(men[0][0]) == 3 and abs(men[1][0]) == 3 and _sign(men[0][0]) != _sign(men[1][0]) and is_light(men[0][1]) == is_light(men[1][1])))
    if bare: return Status(True, "1/2-1/2", "insufficient material", check)
    return Status(False, "", "", check)


def san(pos, m, ms=None):
    """Standard algebraic notation."""
    pos = parse(pos); ms = moves(pos) if ms is None else ms; n = make(pos, m); kind = abs(m.piece)
    if m.castle: s = "O-O" if m.castle == "K" else "O-O-O"
    else:
        if kind == 1: s = ("abcdefgh"[m.frm & 7] + "x") if m.capture else ""
        else:
            s = PIECES[kind - 1]
            others = [o for o in ms if (o.frm != m.frm or o.to != m.to or o.promo != m.promo) and o.piece == m.piece and o.to == m.to]
            if others:
                same_file = any((o.frm & 7) == (m.frm & 7) for o in others); same_rank = any((o.frm >> 3) == (m.frm >> 3) for o in others)
                s += "abcdefgh"[m.frm & 7] if not same_file else str(1 + (m.frm >> 3)) if not same_rank else sq_name(m.frm)
            if m.capture: s += "x"
        s += sq_name(m.to)
        if m.promo: s += "=" + PIECES[m.promo - 1]
    if in_check(n): s += "+" if moves(n) else "#"
    return s


def perft(pos, depth):
    """Leaf count of the legal-move tree (the rules' standard check)."""
    pos = parse(pos)
    if depth <= 0: return 1
    ms = moves(pos)
    if depth == 1: return len(ms)
    return sum(perft(make(pos, m), depth - 1) for m in ms)


# ---------------------------------------------------------------------------------------------------------------- the view
def mirror(pos):
    """The colour-swapped mirror (python-chess Board.mirror): ranks flipped a1 ↔ a8, colours swapped, the other side to move, the
    castling rights and the en-passant square carried across. The move counters are kept."""
    pos = parse(pos); s = [0] * 64
    for i in range(64): s[i ^ 56] = -pos.sq[i]
    swap = {"K": "k", "Q": "q", "k": "K", "q": "Q"}
    cs = {swap[c] for c in pos.castle}
    return Position(tuple(s), "b" if pos.turn == "w" else "w", "".join(k for k in "KQkq" if k in cs), (pos.ep ^ 56) if pos.ep >= 0 else -1, pos.half, pos.full)
def mirror_move(m):
    """A move on a position → the same move on its mirror."""
    return m._replace(frm=m.frm ^ 56, to=m.to ^ 56, piece=-m.piece)


class Look(NamedTuple):
    """How a crystal's frames are drawn (the file's chess block)."""
    side: str = "b"                                                            # chess.playAs: the colour the crystal sees to move
    flat: bool = True                                                          # chess.flatBoard: no checkerboard
    clock: bool = True                                                         # chess.lampClock: the side to move's clock lamp
    mark: bool = True                                                          # chess.lampCrystal: the crystal's quartz beside it
    frame: int = 192                                                           # chess.frame (dataset.frame)
    art96x2: bool = True                                                       # chess.art96x2: the 96-px art doubled


def look_of(crystal_or_model):
    """The Look a crystal file asks for (its chess block). Refuses frames this module does not draw."""
    m = getattr(crystal_or_model, "m", crystal_or_model)
    ch = m.get("chess") or {}
    side = str(ch.get("playAs") or "b"); side = side if side in ("w", "b") else "b"
    F = int(ch.get("frame") or (m.get("dataset") or {}).get("frame") or 192)
    lk = Look(side, bool(ch.get("flatBoard")), ch.get("lampClock", True) is not False, ch.get("lampCrystal", True) is not False, F, bool(ch.get("art96x2")))
    if not lk.art96x2 or lk.frame != 192:
        raise NotImplementedError(f"chess frame {F} (art96x2={lk.art96x2}): only the art96x2 frame (the 96-px art doubled to 192) is supported")
    return lk


def view(pos, side="b"):
    """(the position as the crystal sees it, flipped): mirrored when the side to move is not the crystal's colour."""
    pos = parse(pos)
    return (pos, False) if pos.turn == side else (mirror(pos), True)


# ---------------------------------------------------------------------------------------------------------------- the frame
SIZE, TILE, SPR, ORG, ORGY, EPIN = 96, 10, 8, 8, 8, 1
LAMP = {"turnB": (1, 13, 5, 30), "turnW": (1, 53, 5, 30), "crystalT": (1, 1, 6, 6), "crystalB": (1, 89, 6, 6),
        "castle": {"q": (1, 8, 2, 3), "k": (4, 8, 2, 3), "Q": (1, 85, 2, 3), "K": (4, 85, 2, 3)}}
def _ep_rect(f): return (8 + 10 * f + 3, 88, 4, 5)
# the compact sprite set, 8 × 8: m the silhouette, o the ring, w the white piece's lit pixels (derived from m and o unless given)
SPRITES = {
    "K": {"m": ["...##...", "..####..", "...##...", ".##..##.", "########", "########", ".######.", ".######."],
          "o": ["...##...", "..#..#..", "...##...", ".##..##.", "#......#", "#......#", ".#....#.", ".######."]},
    "Q": {"m": ["........", "#..##..#", "##.##.##", "########", "########", "########", ".######.", ".######."],
          "o": ["#......#", "##.##.##", "#.#..#.#", "#......#", "#......#", ".#....#.", ".#....#.", ".######."],
          "w": ["........", "#..##..#", "##.##.##", "########", "########", "########", ".######.", ".######."]},
    "B": {"m": ["...##...", "..###...", "..##.#..", ".##.###.", ".######.", ".######.", ".######.", ".######."],
          "o": ["...##...", "..#..#..", "..#..#..", ".#..#.#.", ".#.#..#.", ".#....#.", ".#....#.", ".######."]},
    "N": {"m": ["..######", ".#######", "########", "########", "########", ".##.####", "....####", "...#####"],
          "o": ["..######", ".#.....#", "#..#...#", "#......#", "#..##..#", ".##.#..#", "....#..#", "...#####"]},
    "R": {"m": [".#.##.#.", ".######.", ".######.", ".######.", "..####..", "..####..", "..####..", "..####.."],
          "o": ["##.##.##", "#.#..#.#", "#......#", ".#....#.", ".#....#.", ".#....#.", ".#....#.", ".######."]},
    "P": {"m": ["........", "...##...", "..####..", "..####..", "...##...", "..####..", ".######.", ".######."],
          "o": ["........", "...##...", "..#..#..", "..#..#..", "...##...", "..#..#..", ".#....#.", ".######."]},
}
for _S in SPRITES.values():
    _n = len(_S["m"])
    def _edge(y, x, S=_S, n=_n): return any(not (0 <= y + dy < n and 0 <= x + dx < n) or S["m"][y + dy][x + dx] != "#" for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    if "w" not in _S:
        _S["w"] = ["".join("#" if c == "#" and not (_S["o"][y][x] == "#" and not _edge(y, x)) else "." for x, c in enumerate(row)) for y, row in enumerate(_S["m"])]
del _S, _n


def _sprite_px(S, solid, light, y, x):                                         # 1 / 0 on the sprite, −1 where the square shows
    m, o = S["m"][y][x] == "#", S["o"][y][x] == "#"
    if not m and not o: return -1
    if solid: return (0 if o else (1 if m else -1)) if light else ((1 if S["w"][y][x] == "#" else 0) if m else -1)
    return ((0 if S["w"][y][x] == "#" else 1) if m else -1) if light else (1 if o else -1)


def _rect(v, x, y, w, h, val): v[y:y + h, x:x + w] = val
def _quartz(v, x0, y0, w, h):                                                  # a double-terminated crystal
    cy = y0 + (h - 1) / 2; tip = max(2, math.floor(w * 0.30 + 0.5)); hm = (h - 2) / 2
    for x in range(x0, x0 + w):
        rx = x - x0
        hh = hm * (rx + 1) / tip if rx < tip else hm * (w - rx) / tip if rx >= w - tip else hm
        for y in range(y0, y0 + h):
            if abs(y - cy) <= hh + 0.001: v[y, x] = 1.0


def render96(pos, look=Look(), legal=None):
    """The 96 × 96 picture of a position exactly as given (no mirroring), float32 0 / 1: the reference renderer on the compact layout with
    the crystal's switches (no seed noise, the clock lamp, the quartz mark on the side to move)."""
    pos = parse(pos); v = np.zeros((SIZE, SIZE), dtype=np.float32)
    for s in range(64):
        x0, y0 = ORG + TILE * (s & 7), ORGY + TILE * (7 - (s >> 3)); light = (not look.flat) and is_light(s)
        if light: v[y0:y0 + TILE, x0:x0 + TILE] = 1.0                        # a lit square: fill 64 of 64 on the Bayer matrix, all on
        p = pos.sq[s]
        if p:
            S = SPRITES[PIECES[abs(p) - 1]]; o = (TILE - SPR) // 2
            for y in range(SPR):
                for x in range(SPR):
                    q = _sprite_px(S, p > 0, light, y, x)
                    if q >= 0: v[y0 + o + y, x0 + o + x] = q
    _rect(v, *LAMP["turnW"], 0.0); _rect(v, *LAMP["turnB"], 0.0)
    if look.clock: _rect(v, *(LAMP["turnW"] if pos.turn == "w" else LAMP["turnB"]), 1.0)
    for k in "KQkq": _rect(v, *LAMP["castle"][k], 1.0 if k in pos.castle else 0.0)
    if pos.ep >= 0 and any(m.ep for m in (moves(pos) if legal is None else legal)):
        x, y, w, h = _ep_rect(pos.ep & 7); _rect(v, x, y, w, h, 0.0); _rect(v, x + EPIN, y + EPIN, w - 2 * EPIN, h - 2 * EPIN, 1.0)
    if look.mark: _quartz(v, *(LAMP["crystalT"] if pos.turn == "b" else LAMP["crystalB"]))
    return v


def render(pos, look=Look()):
    """The 192 × 192 DMD frame of a position exactly as given (no mirroring): the 96-px art, every pixel doubled."""
    return np.repeat(np.repeat(render96(pos, look), 2, axis=0), 2, axis=1)


def frame(pos, look=Look()):
    """The frame the crystal is shown for a position: its view (mirrored when the other colour is to move), 192 × 192 float32 0 / 1."""
    return render(view(pos, look.side)[0], look)


def frames(positions, look=Look()):
    """A batch of positions → float32 [B, 192, 192], ready for crystal.run."""
    return torch.from_numpy(np.stack([frame(p, look) for p in positions]))


# ---------------------------------------------------------------------------------------------------------------- the move map
# 64 × 64 cells, cell = from × 64 + to in the VIEW's squares; cells 16/1024 of the lit window apart, each 8/1024 of it square.
GRID, PITCH, HALF = 64, 16 / 1024, 4 / 1024
def cell_of(frm, to): return frm * 64 + to
def cell_move(cell): return cell // 64, cell % 64                             # (from, to) in the view's squares


def legal_cells(pos, side="b"):
    """{cell: move} over the legal moves, the cell in the crystal's view and the move on the real board, in the reference cell order
    (a promotion's four moves share a cell and read as the queen)."""
    pos = parse(pos); v, flip = view(pos, side); out = {}
    for m in moves(v):
        c = cell_of(m.frm, m.to)
        if c not in out or m.promo == PROMO_Q: out[c] = mirror_move(m) if flip else m
    return out


def _r(x): return math.floor(x + 0.5)                                         # JavaScript's Math.round, as the simulator's detector rectangles
def bands(N, win_frac=1.0):
    """The cells' bands on the N × N grid: cell d covers samples [lo[d], lo[d] + side) on each axis. Rows = from, columns = to."""
    w = N * float(win_frac); c = (N - 1) / 2
    return [_r(c + ((d - (GRID - 1) / 2) * PITCH - HALF) * w) for d in range(GRID)], _r(2 * HALF * w)
def _band_matrix(crystal):
    key = (crystal.N, crystal.win_frac, str(crystal.dev))
    if getattr(crystal, "_chess_key", None) != key:
        lo, side = bands(crystal.N, crystal.win_frac); A = torch.zeros(crystal.N, GRID)
        for d, x0 in enumerate(lo):
            a, b = max(0, x0), min(crystal.N, x0 + side)
            if b > a: A[a:b, d] = 1.0
        crystal._chess_A = A.to(crystal.dev); crystal._chess_key = key
    return crystal._chess_A
def move_powers(crystal, E):
    """The light in each of the 4,096 cells from the exit field E [B, N, N] (or [N, N]) → float32 [B, 4096] (Aᵀ·|E|²·A)."""
    if E.dim() == 2: E = E.unsqueeze(0)
    I = (E.real * E.real + E.imag * E.imag).float(); A = _band_matrix(crystal)
    return torch.einsum("yf,byx,xt->bft", A, I, A).reshape(I.shape[0], GRID * GRID)
def move_map(crystal, E):
    """The exit face's move map as a picture [B, 64, 64]: row = from-square, column = to-square (a1 = 0 … h8 = 63, as the crystal sees the board)."""
    return move_powers(crystal, E).reshape(-1, GRID, GRID)


# ---------------------------------------------------------------------------------------------------------------- the value readout
def value_cells(crystal_or_model):
    """The readout's three groups of cells (win, draw, loss) from the file (chess.value, else tr.value): [first, last] runs of the
    diagonal (cell 65·s), or {"block": [r0, c0, r1, c1]}. None when the crystal has no readout."""
    m = getattr(crystal_or_model, "m", crystal_or_model)
    v = (m.get("chess") or {}).get("value") or (m.get("tr") or {}).get("value")
    if not isinstance(v, dict): return None
    gs = []
    for k in ("win", "draw", "loss"):
        r = v.get(k)
        if isinstance(r, dict) and "block" in r:
            r0, c0, r1, c1 = (int(x) for x in r["block"]); gs.append([a * GRID + b for a in range(r0, r1 + 1) for b in range(c0, c1 + 1)])
        elif isinstance(r, (list, tuple)) and len(r) == 2:
            lo, hi = int(r[0]), int(r[1])
            if not 0 <= lo <= hi < GRID: raise ValueError(f"value.{k} = {r}: squares run 0..63")
            gs.append([(GRID + 1) * s for s in range(lo, hi + 1)])
        else: return None
    flat = [c for g in gs for c in g]
    if len(set(flat)) != len(flat): raise ValueError(f"value groups overlap: {v}")
    return gs
def value_of(crystal, p):
    """[B, 4096] cell light → [B, 3] (W, D, L) shares of the side to move (the crystal's view; mirroring keeps the side to move)."""
    gs = value_cells(crystal)
    if gs is None: raise ValueError("this crystal has no value readout (chess.value / tr.value)")
    p = torch.as_tensor(p, dtype=torch.float32)
    if p.dim() == 1: p = p.unsqueeze(0)
    G = torch.zeros(3, GRID * GRID, device=p.device)
    for i, g in enumerate(gs): G[i, g] = 1.0
    V = p @ G.t()
    return V / V.sum(1, keepdim=True).clamp(min=1e-30)
def value(crystal, E):
    """The win / draw / loss readout of an exit field [B, N, N] → [B, 3], for the side to move in the position shown."""
    return value_of(crystal, move_powers(crystal, E))


# ---------------------------------------------------------------------------------------------------------------- reading a move
def _ranked(p, lc):
    """[(move, light)] over the legal cells, brightest first; a tie goes to the lower cell (crystal_ranked's stable sort)."""
    return [(lc[c], float(p[c])) for c in sorted(lc, key=lambda c: -float(p[c]))]


def move_scores(crystal, E_exit, pos):
    """Every legal move with its cell's light, brightest first: [(move, light)]. E_exit: one position's exit field [N, N] or [1, N, N]."""
    pos = parse(pos); p = move_powers(crystal, E_exit)[0].cpu().numpy()
    return _ranked(p, legal_cells(pos, look_of(crystal).side))


def read_move(p, pos, side="b", temp=0.0, top_k=0, rng=None):
    """The move the light names. p: the 4,096 cells' light.
    temp = 0 (the default): the brightest legal cell, ties to the lower cell. temp > 0: sample among the top_k brightest legal cells
    (all if 0) with odds ∝ (light / brightest)^(1 / temp) — the file's chess.mmPlayT / mmTopK, which the simulator's game uses for variety;
    the published scores use temp = 0."""
    p = np.asarray(p, dtype=np.float64); lc = legal_cells(pos, side)
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


def read(crystal, positions, batch=8):
    """Each position through the crystal: [(light [4096] float32 numpy, wdl (w, d, l) or None)], `batch` positions a forward pass."""
    look = look_of(crystal); positions = [parse(p) for p in positions]; has_v = value_cells(crystal) is not None; out = []
    for i in range(0, len(positions), batch):
        E, _ = crystal.run(frames(positions[i:i + batch], look))
        P = move_powers(crystal, E); V = value_of(crystal, P).cpu().numpy() if has_v else None; P = P.cpu().numpy()
        for j in range(P.shape[0]): out.append((P[j], tuple(float(x) for x in V[j]) if has_v else None))
        del E
    return out


# ---------------------------------------------------------------------------------------------------------------- the look-ahead
# Game ends INSIDE the tree are scored as the trainer's think check scored them — python-chess's Board.outcome(claim_draw=True):
# checkmate, its insufficient-material rule, stalemate, the 50-move claim (including a move that reaches it) and threefold repetition
# along the line from the root.
def _insufficient(pos, col):                                                   # python-chess has_insufficient_material(colour)
    b = pos.sq; mine = [p for p in b if p and _sign(p) == col]
    if any(abs(p) in (1, 4, 5) for p in mine): return False
    if any(abs(p) == 2 for p in mine):
        return len(mine) <= 2 and not any(p and _sign(p) == -col and abs(p) not in (5, 6) for p in b)
    if any(abs(p) == 3 for p in mine):
        bish = [i for i, p in enumerate(b) if abs(p) == 3]
        same = all(is_light(i) for i in bish) or not any(is_light(i) for i in bish)
        return same and not any(abs(p) in (1, 2) for p in b)
    return True
def _zeroing(m): return abs(m.piece) == 1 or m.capture
def _outcome(pos, me, line=()):
    """None while the game goes on; else our score there (1 won, ½ drawn, 0 lost) for colour `me`. `line`: the positions from the
    think's root to this one's parent, oldest first, with the moves between ((pos, move) pairs)."""
    ms = moves(pos)
    if not ms and in_check(pos): return 1.0 if pos.turn != me else 0.0
    if _insufficient(pos, 1) and _insufficient(pos, -1): return 0.5
    if not ms: return 0.5
    if pos.half >= 100: return 0.5
    if pos.half >= 99:
        for m in ms:
            if not _zeroing(m):
                n = make(pos, m)
                if n.half >= 100 and moves(n): return 0.5
    if len(line) >= 3:                                                         # threefold: repetitions back to the last irreversible move
        k0 = key(pos); seen = {k0: 1}
        for prev, m in reversed(line):
            if _zeroing(m) or make(prev, m).castle != prev.castle or (prev.ep >= 0 and any(x.ep for x in moves(prev))): break
            kk = key(prev); seen[kk] = seen.get(kk, 0) + 1
        if seen[k0] >= 3: return 0.5
        for m in ms:
            if seen.get(key(make(pos, m)), 0) >= 2: return 0.5
    return None


def _es(wdl, ours): w, d, l = wdl; return (w + 0.5 * d) if ours else (l + 0.5 * d)


def look_ahead(crystal, positions, ranked, widths=(3, 5), batch=8):
    """The look-ahead (the training code's 2-ply K × J; other widths work the same way). For each position (its side
    to move = 'us') and its ranked legal moves: our top widths[0] moves are each played; at every following ply the board is shown to
    the crystal (mirrored when the other colour is to move) and its top widths[d] legal moves are played; after the last ply the
    crystal's value readout scores the board — W + ½D when we are to move, the opponent's L + ½D when they are. Our plies take the best
    child, theirs the worst; a move that ends the game scores 1 / ½ / 0 directly. Widths (3,) is 1-ply, (3, 5) the published 2-ply.
    Returns per position (pick index into ranked, [expected score per candidate], forward passes the whole batch used), or None where it did not think
    (fewer than two legal moves). Strictly better only: a tie keeps the brighter move."""
    W = [int(x) for x in widths]; K = W[0]; out = [None] * len(positions); es = [None] * len(positions)
    val, kids, ours, order, root, level, nid = {}, {}, {}, [], {}, [], [0]
    def new(): nid[0] += 1; return nid[0]
    for k, (b, rk) in enumerate(zip(positions, ranked)):
        cand = rk[:K]
        if len(cand) < 2: continue
        es[k] = [None] * len(cand); me = b.turn
        for i, (m, _) in enumerate(cand):
            nb = make(b, m); o = _outcome(nb, me, ((b, m),))
            if o is not None: es[k][i] = o
            else: n_ = new(); root[(k, i)] = n_; level.append((n_, nb, me, ((b, m),)))
    passes = 0
    for d in range(1, len(W) + 1):
        if not level: break
        rd = read(crystal, [b for _, b, _, _ in level], batch); passes += len(level)
        if d == len(W):                                                        # the last ply: the value, for whoever is to move
            for (n_, b, me, _), (_, wdl) in zip(level, rd):
                if wdl is None: raise ValueError("this crystal has no value readout: it cannot think")
                val[n_] = _es(wdl, b.turn == me)
            break
        side = look_of(crystal).side; nxt = []
        for (n_, b, me, line), (p, _) in zip(level, rd):
            ours[n_] = b.turn == me; kids[n_] = []; order.append(n_)
            for m, _ in _ranked(p, legal_cells(b, side))[:W[d]]:
                b2 = make(b, m); c_ = new(); kids[n_].append(c_); l2 = line + ((b, m),)
                o = _outcome(b2, me, l2)
                if o is not None: val[c_] = o
                else: nxt.append((c_, b2, me, l2))
        level = nxt
    for n_ in reversed(order):                                                 # back up, deepest first
        vs = [val[x] for x in kids[n_] if x in val]
        val[n_] = (max(vs) if ours[n_] else min(vs)) if vs else 0.5
    for k in range(len(positions)):
        if es[k] is None: continue
        for i in range(len(es[k])):
            if es[k][i] is None: es[k][i] = val.get(root[(k, i)], 0.5)
        best = 0
        for i in range(1, len(es[k])):
            if es[k][i] > es[k][best]: best = i
        out[k] = (best, es[k], passes)
    return out


# ---------------------------------------------------------------------------------------------------------------- playing
class Choice(NamedTuple):
    move: object                                                               # the Move played (None: no legal move)
    scores: list                                                               # [(move, light)], every legal move, brightest first
    wdl: tuple                                                                 # (w, d, l) for the side to move, or None (no readout)
    think: dict                                                                # None, or {"widths", "cand": [uci], "es": [expected score], "changed", "passes"}


def best_moves(crystal, positions, think=None, temp=0.0, top_k=0, rng=None, batch=8):
    """Each position (FEN or Position) through the crystal and its move read: [Choice]. One forward pass a position, plus the
    look-ahead's when `think` is given: an int K (1-ply) or widths (K, J, …) — (3, 5) is the published 2-ply, ≤ 1 + 3 + 15 passes a
    position. temp / top_k sample the move as the simulator's game does; they apply only without `think`."""
    positions = [parse(p) for p in positions]; look = look_of(crystal); rd = read(crystal, positions, batch); out = []
    ranked = [_ranked(p, legal_cells(pos, look.side)) for pos, (p, _) in zip(positions, rd)]
    th = None
    if think:
        widths = (int(think),) if isinstance(think, int) else tuple(int(x) for x in think)
        th = look_ahead(crystal, positions, ranked, widths, batch)
    for k, (pos, (p, wdl), rk) in enumerate(zip(positions, rd, ranked)):
        if not rk: out.append(Choice(None, [], wdl, None)); continue
        if th is not None and th[k] is not None:
            best, es, passes = th[k]; K = len(es)
            out.append(Choice(rk[best][0], rk, wdl, {"widths": widths, "cand": [uci(m) for m, _ in rk[:K]], "es": es, "changed": best != 0, "passes": passes}))
        else:
            out.append(Choice(read_move(p, pos, look.side, temp, top_k, rng) if th is None else rk[0][0], rk, wdl, None))
    return out


def best_move(crystal, pos, think=None, temp=0.0, top_k=0, rng=None):
    """Choice(move, scores, wdl, think) for one position: the crystal's move (None if it has no legal move), every legal move's light
    brightest first, its win / draw / loss readout for the side to move, and what the look-ahead saw (think=K or (K, J))."""
    return best_moves(crystal, [pos], think, temp, top_k, rng)[0]
