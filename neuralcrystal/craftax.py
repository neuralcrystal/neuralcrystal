"""The Craftax crystal's problem: the two pictures it sees each step, and the move and next memory it answers with.

Craftax (Matthews et al., 2024) is a 2D survival research benchmark. This module doesn't run the game: give it a game state and it draws
the crystal's two input pictures, runs the light, and reads the move. The crystal is recurrent: it carries a memory of 1,024 numbers
(4 bits each) from one step to the next, drawn into its second picture, and writes the next memory as light.

    import neuralcrystal as nc
    from neuralcrystal import craftax
    crystal = nc.load("craftax-crystal.safetensors")
    agent = craftax.Agent(crystal)                 # holds the memory; agent.reset() for a new game
    action = agent.act(state)                      # a Craftax game state → the move (0 … 42; craftax.ACTIONS names them)

`state` is a JAX Craftax EnvState (pip install craftax; one world, not a batch) or the same fields as a dict in the browser engine's JSON
form (craftax.state_dict turns one into the other). Pass Craftax's action mask as mask= to keep the crystal to moves the game allows.

The board picture is a port of the game page's frame.js, itself checked pixel for pixel against the frame drawer the crystal was
trained with: the 9 × 11 tiles around the player as 16 × 16 symbols (the creature, else the item, else the block; unlit tiles dark; the
player turned to its facing; fighters with a health bar), the status pads under them, and the player's tile marked with its needs and its
pickaxe.
"""
import json
import math
import os

import numpy as np
import torch

from .chess import move_powers

ACTIONS = ["noop", "left", "right", "up", "down", "do", "sleep", "place stone", "place table", "place furnace", "place plant",
           "make wood pickaxe", "make stone pickaxe", "make iron pickaxe", "make wood sword", "make stone sword", "make iron sword", "rest",
           "descend", "ascend", "make diamond pickaxe", "make diamond sword", "make iron armor", "make diamond armor", "shoot arrow",
           "make arrow", "cast fireball", "cast iceball", "place torch", "drink red potion", "drink green potion", "drink blue potion",
           "drink pink potion", "drink cyan potion", "drink yellow potion", "read book", "enchant sword", "enchant armor", "make torch",
           "level up dexterity", "level up strength", "level up intelligence", "enchant bow"]

VISIBLE = 12                     # light (0 … 255) above this is lit: the JAX game's light > 0.05
CLASS_FIELD = {"passive": "passive", "melee": "melee", "ranged": "ranged", "mob projectile": "mob_projectiles", "your projectile": "player_projectiles"}

_ATLAS = None


def atlas():
    """The symbols and placements the board picture is drawn from (craftax_frame_atlas.json, the game page's frame-atlas.json)."""
    global _ATLAS
    if _ATLAS is None:
        with open(os.path.join(os.path.dirname(__file__), "craftax_frame_atlas.json")) as f: A = json.load(f)
        bits = lambda s, n: np.unpackbits(np.frombuffer(__import__("base64").b64decode(s), np.uint8))[:n]
        TL = A["tile"]
        A["glyph_bits"] = bits(A["glyphs"], A["n_glyphs"] * TL * TL).reshape(A["n_glyphs"], TL, TL)
        A["static_bits"] = bits(A["static"], A["frame"] * A["frame"])
        A["pads_np"] = [[np.asarray(p, np.int64) for p in pads] for pads in A["pads"]]
        _ATLAS = A
    return _ATLAS


def round_even(x):
    """Round halves to even, as JAX does (Python's round does the same)."""
    return int(round(x))


# ================================================================================================================ the game state

def state_dict(st):
    """A JAX Craftax EnvState (one world) → the fields the board picture reads, in the browser engine's JSON form. Only numpy is
    needed: the arrays are read with np.asarray, so the state can come from JAX or from a saved copy."""
    A = np.asarray; lv = int(A(st.player_level)); inv = st.inventory
    def mobs(m, dirs=None):
        mask = A(m.mask)[lv]; out = []
        for i in np.nonzero(mask)[0]:
            e = {"type": int(A(m.type_id)[lv, i]), "pos": A(m.position)[lv, i].tolist(), "hp": float(A(m.health)[lv, i])}
            if dirs is not None: e["dir"] = A(dirs)[lv, i].tolist()
            out.append(e)
        return out
    light = np.clip(A(st.light_map)[lv] * 255.0, 0, 255).astype(np.uint8)
    return {"level": lv, "map": A(st.map)[lv].ravel(), "items": A(st.item_map)[lv].ravel(), "light": light.ravel(), "size": [48, 48],
            "pos": A(st.player_position).tolist(), "dir": int(A(st.player_direction)), "daylight": float(A(st.light_level)),
            "health": float(A(st.player_health)), "food": int(A(st.player_food)), "drink": int(A(st.player_drink)), "energy": int(A(st.player_energy)),
            "mana": int(A(st.player_mana)), "sleeping": bool(A(st.is_sleeping)), "resting": bool(A(st.is_resting)), "xp": int(A(st.player_xp)),
            "dexterity": int(A(st.player_dexterity)), "strength": int(A(st.player_strength)), "intelligence": int(A(st.player_intelligence)),
            "inventory": {**{k: int(A(getattr(inv, k))) for k in ("wood", "stone", "coal", "iron", "diamond", "sapling", "pickaxe", "sword", "bow", "arrows", "torches", "ruby", "sapphire", "books")},
                          "armour": A(inv.armour).tolist(), "potions": A(inv.potions).tolist()},
            "sword_enchant": int(A(st.sword_enchantment)), "bow_enchant": int(A(st.bow_enchantment)), "armour_enchant": A(st.armour_enchantments).tolist(),
            "spells": [bool(x) for x in A(st.learned_spells)], "boss_spawn_turns": max(0, int(A(st.boss_timesteps_to_spawn_this_round))),
            "monsters_killed": int(A(st.monsters_killed)[lv]), "melee": mobs(st.melee_mobs), "ranged": mobs(st.ranged_mobs), "passive": mobs(st.passive_mobs),
            "mob_projectiles": mobs(st.mob_projectiles, st.mob_projectile_directions), "player_projectiles": mobs(st.player_projectiles, st.player_projectile_directions)}


def _as_dict(state): return state if isinstance(state, dict) else state_dict(state)


# ================================================================================================================ the board picture

def _hud_on(A, s, mode):
    """Which of each status row's 9 pads are lit, in the atlas's hud order."""
    inv, lv = s["inventory"], s["level"]
    thermo = lambda n: [k < min(9, n) for k in range(9)]
    root = lambda v: thermo(round_even(math.sqrt(v)))
    hot = lambda i: [k == i for k in range(9)]
    v = {"health": thermo(round_even(s["health"])), "food": thermo(s["food"]), "drink": thermo(s["drink"]), "energy": thermo(s["energy"]),
         "mana": thermo(s["mana"]), "xp": thermo(s["xp"]), "dexterity": thermo(s["dexterity"]), "strength": thermo(s["strength"]),
         "intelligence": thermo(s["intelligence"]), "floor": hot(lv), "light": thermo(round_even(9 * s["daylight"]) if lv == 0 else 4),
         "pickaxe tier": thermo(inv["pickaxe"]), "sword tier": thermo(inv["sword"]), "bow": thermo(inv["bow"]), "books": thermo(inv["books"]),
         "sword ench.": thermo(s["sword_enchant"]), "bow ench.": thermo(s["bow_enchant"]),
         "facing": hot([3, 5, 1, 7][min(3, max(0, s["dir"] - 1))]), "mode": hot([0, 2, 6, 8][mode])}
    for n in ("wood", "stone", "coal", "iron", "diamond", "sapling", "ruby", "sapphire", "arrows", "torches"): v[n] = root(inv[n])
    for i, c in enumerate(("red", "green", "blue", "pink", "cyan", "yellow")): v[f"potion {c}"] = root(inv["potions"][i])
    for i, n in enumerate(("helmet", "chestplate", "pants", "boots")): v[n] = thermo(inv["armour"][i])
    for i, n in enumerate(("helmet ench.", "chest ench.", "pants ench.", "boots ench.")): v[n] = thermo(s["armour_enchant"][i])
    boss = lv == 8 and len(s["melee"]) == 0 and len(s["ranged"]) == 0 and s["boss_spawn_turns"] <= 0
    v["flags"] = [bool(s["sleeping"]), bool(s["resting"]), bool(s["spells"][0]), bool(s["spells"][1]), s["monsters_killed"] >= 8, boss, False, False, False]
    return [v[n] for n in A["hud"]]


def _view_tiles(A, s):
    """The 9 × 11 tiles' symbols around the player, and each tile's health bar (0: none)."""
    H, W = A["view"]; NB = A["n_blocks"]; MS = s["size"][0]; cy, cx = H >> 1, W >> 1; py, px = s["pos"]
    mp, items, light = np.asarray(s["map"]).ravel(), np.asarray(s["items"]).ravel(), np.asarray(s["light"]).ravel()
    oob = A["blocks"].index("out of bounds")
    tile = np.zeros(H * W, np.int64); hp = np.zeros(H * W, np.float64); lit = np.zeros(H * W, bool); ent = np.full(H * W, -1, np.int64)
    for r in range(H):
        for c in range(W):
            y, x, k = py - cy + r, px - cx + c, r * W + c
            inside = 0 <= y < MS and 0 <= x < MS
            blk = int(mp[y * MS + x]) if inside else oob; item = int(items[y * MS + x]) if inside else 0
            tile[k] = NB + item - 1 if item > 0 else blk
            lit[k] = inside and int(light[y * MS + x]) > VISIBLE
    for ci, cls in enumerate(A["classes"]):
        for m in s.get(CLASS_FIELD[cls]) or []:
            r, c = m["pos"][0] - py + cy, m["pos"][1] - px + cx
            if r < 0 or r >= H or c < 0 or c >= W: continue
            g = A["cls_base"][ci][m["type"]]
            if "dir" in m:
                dy, dx = m["dir"]; g += 0 if dx < 0 else 1 if dx > 0 else 2 if dy < 0 else 3
            ent[r * W + c] = g; hp[r * W + c] = m["hp"] if cls in ("melee", "ranged") else 0
    for k in range(H * W):
        if ent[k] >= 0: tile[k] = ent[k]
        if not lit[k]: tile[k] = A["dark"]
        if not (lit[k] and ent[k] >= 0): hp[k] = 0
    ck = cy * W + cx; hp[ck] = 0; tile[ck] = A["player0"] + min(3, max(0, s["dir"] - 1))
    return tile, hp


# the player's tile marked with its needs and its pickaxe (the crystal was trained with this mark): each need at 3 or less hollows part
# of the body (thirsty: head; hungry: tummy; tired: arms; hurt: legs and feet), and the pickaxe tier lights 3 × 3 corners 1 px in:
# top right (wood), top left (stone), bottom left (iron), bottom right (diamond)
TILE_Y, TILE_X = 64, 80
HOLES = [[(5, 7), (5, 8), (6, 7), (6, 8)], [(8, 7), (8, 8), (9, 7), (9, 8)], [(8, 4), (8, 5), (9, 4), (8, 10), (8, 11), (9, 11)],
         [(10, 6), (10, 9), (11, 5), (11, 6), (11, 9), (11, 10)]]
CORNERS = [(1, 12), (1, 1), (12, 1), (12, 12)]


def _stamp_player(fr, s):
    needs = [s["drink"] <= 3, s["food"] <= 3, s["energy"] <= 3, round_even(s["health"]) <= 3]
    for on, holes in zip(needs, HOLES):
        if on:
            for y, x in holes: fr[TILE_Y + y, TILE_X + x] = 0
    for y0, x0 in CORNERS[:min(4, max(0, s["inventory"]["pickaxe"]))]: fr[TILE_Y + y0:TILE_Y + y0 + 3, TILE_X + x0:TILE_X + x0 + 3] = 1


def frame(state, mode=0, stamp=True):
    """The 192 × 192 board picture for a game state (uint8, 0 dark mirror, 1 lit), memory cells blank. mode: the status pad for the
    mode (0 … 3; the game page draws 0). stamp: the player's tile marked with its needs and pickaxe, as the crystal was trained."""
    A = atlas(); s = _as_dict(state); F, TL = A["frame"], A["tile"]; H, W = A["view"]
    tile, hp = _view_tiles(A, s)
    fr = A["static_bits"].copy().reshape(F, F)
    for r in range(H):
        for c in range(W):
            k = r * W + c; g = A["glyph_bits"][tile[k]].copy()
            if hp[k] > 0:
                n = min(TL, max(0, math.ceil(hp[k] / 2))); g[TL - 2:] = 0; g[TL - 1, :n] = 1
            fr[r * TL:(r + 1) * TL, c * TL:(c + 1) * TL] = g
    flat = fr.reshape(-1)
    for j, on in enumerate(_hud_on(A, s, mode)):
        for k, o in enumerate(on):
            if o: flat[A["pads_np"][j][k]] = 1
    if stamp: _stamp_player(fr, s)
    return fr


# ================================================================================================================ the crystal

def _layout(crystal):
    cx = (crystal.m.get("tr") or {}).get("craftax")
    if not cx or not cx.get("e2eHead") or not cx.get("split"): raise ValueError("not a Craftax crystal (no tr.craftax with a split layout and its heads)")
    if getattr(crystal, "_craftax", None) is None:
        sp = cx["split"]; mA, mB = np.asarray(sp["mapA"], np.int64), np.asarray(sp["mapB"], np.int64)
        crystal._craftax = {"F": int(cx["frame"]), "units": int(cx["units"]), "mapA": mA, "mapB": mB, "mem": np.asarray(cx["mem"], np.int64),
                            "act": [np.atleast_1d(np.asarray(c, np.int64)) for c in cx["cells"]["act"]],
                            "z": np.asarray(cx["cells"]["z"], np.int64), "hc": np.asarray(cx["cells"]["hc"], np.int64),
                            "gA": np.asarray(cx["e2eHead"]["gA"], np.float32), "al": np.asarray(cx["e2eHead"]["al"], np.float32),
                            "be": np.asarray(cx["e2eHead"]["be"], np.float32)}
    return crystal._craftax


def pictures(crystal, board, memory):
    """The two input pictures from the board picture [F, F] and the memory (1,024 numbers in 0 … 1, 4-bit levels): the first input
    head's picture (the board and its key fields, gathered by the crystal's mapA) and the second's (the slower fields gathered by mapB,
    with each memory number's 4 bits, high bit first, written onto its mirrors). Returns two float32 [F, F] arrays."""
    L = _layout(crystal); F = L["F"]; fr = np.asarray(board, np.float32).reshape(-1)
    a = np.where(L["mapA"] >= 0, fr[np.clip(L["mapA"], 0, None)], 0).astype(np.float32)
    b = np.where(L["mapB"] >= 0, fr[np.clip(L["mapB"], 0, None)], 0).astype(np.float32)
    q = np.round(np.clip(np.asarray(memory, np.float32), 0, 1) * 15).astype(np.int64)
    bits = np.stack([(q >> s) & 1 for s in (3, 2, 1, 0)], -1).astype(np.float32)          # [units, 4]
    mem = L["mem"]
    if mem.ndim == 3: bits = np.broadcast_to(bits[..., None], mem.shape)                 # a bit drawn on a block of mirrors
    b[mem.reshape(-1)] = bits.reshape(-1)
    return a.reshape(F, F), b.reshape(F, F)


@torch.no_grad()
def step(crystal, first, second):
    """One step of the light: the two input pictures ([F, F] or [B, F, F]) → (the 43 actions' scores [B, 43], the memory's update
    gates z [B, 1,024] and new values hc [B, 1,024]). A move's chance is the softmax of its score (over the moves allowed); the next
    memory is h + z·(hc − h), rounded to 4 bits (Agent does this)."""
    L = _layout(crystal)
    E_act, E_mem = crystal.run_split(first, second)
    p1, p2 = move_powers(crystal, E_act), move_powers(crystal, E_mem)             # the light on the 64 × 64 grid of cells
    dev = p1.device; gA, al, be = (torch.tensor(L[k], device=dev) for k in ("gA", "al", "be"))
    lq2 = torch.log(p2 + 1e-12)
    scores = torch.stack([torch.log(p1[:, torch.as_tensor(c, device=dev)].sum(-1) + 1e-12) if len(c) > 1 else torch.log(p1[:, int(c[0])] + 1e-12) for c in L["act"]], 1) + gA
    lz, lh = lq2[:, torch.as_tensor(L["z"], device=dev)], lq2[:, torch.as_tensor(L["hc"], device=dev)]
    mean = torch.cat([lz, lh], 1).mean(1, keepdim=True)
    return scores, torch.sigmoid(al[0] * (lz - mean) + be[0]), torch.sigmoid(al[1] * (lh - mean) + be[1])


class Agent:
    """The crystal playing Craftax: it keeps the memory from step to step. act(state) → the move; reset() at a new game."""

    def __init__(self, crystal, temperature=1.0, seed=None):
        self.crystal = crystal; self.temperature = temperature; self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.memory = np.zeros(_layout(self.crystal)["units"], np.float32)        # a new game starts with every number at 0

    def act(self, state, mask=None, greedy=False):
        """The move for this game state (0 … 42), sampled at the agent's temperature (as the game page plays) or the most likely with
        greedy=True. mask: Craftax's action mask (43 bools), else every move is allowed. Asleep or resting, the game takes no move:
        this returns 0 and the memory stays as it was. After the call, .probs holds the 43 moves' chances and .scores their scores."""
        s = _as_dict(state)
        first, second = pictures(self.crystal, frame(s), self.memory)
        scores, z, hc = step(self.crystal, first, second)
        scores, z, hc = scores[0].cpu().numpy().astype(np.float64), z[0].cpu().numpy(), hc[0].cpu().numpy()
        allowed = np.ones(43, bool) if mask is None else np.asarray(mask, bool)
        logits = np.where(allowed, scores / max(self.temperature, 1e-6), -np.inf)
        p = np.exp(logits - logits.max()); p /= p.sum(); self.scores, self.probs = scores, p
        if s["sleeping"] or s["resting"]: return 0
        h = np.clip(self.memory + z * (hc - self.memory), 0, 1); self.memory = (np.round(h * 15) / 15).astype(np.float32)
        return int(np.argmax(p)) if greedy else int(self.rng.choice(43, p=p))
