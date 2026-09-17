"""Bitboard core for 2048.

Board representation
--------------------
A board is a single Python ``int`` holding 16 nibbles (4 bits each).
Cell ``i`` (row-major, ``i = 4*row + col``) lives in bits ``[4*i, 4*i+3]``.

A nibble stores the *exponent* of the tile, not the tile value:

    0 -> empty      1 -> 2      2 -> 4      3 -> 8   ...   11 -> 2048
    12 -> 4096      13 -> 8192  14 -> 16384 15 -> 32768

Storing exponents is what makes the whole board fit in 64 bits, which in turn
lets a whole row (4 cells = 16 bits) be used directly as an index into a
precomputed move table. A move is then 4 table lookups plus some shifts, with
zero object allocation -- which matters a great deal in pure Python.

Rows are indexed ``r`` = 0..3 top to bottom; row ``r`` occupies bits
``[16*r, 16*r+15]``.

Everything in this module is a free function taking and returning ints. There
is no rendering, no I/O and no game object here on purpose: the training loop
must never pay for anything it does not use.
"""

from __future__ import annotations

from array import array
from random import Random

__all__ = [
    "UP", "DOWN", "LEFT", "RIGHT", "ACTIONS", "ACTION_NAMES",
    "move", "move_no_score", "transpose",
    "spawn_tile", "random_spawn", "spawn_positions",
    "empty_count", "max_tile_exp", "max_tile", "is_game_over",
    "legal_actions", "new_game", "to_list", "from_list",
    "board_to_rows", "render",
]

# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = (UP, RIGHT, DOWN, LEFT)
ACTION_NAMES = ("up", "right", "down", "left")

ROW_MASK = 0xFFFF
FULL_MASK = 0xFFFFFFFFFFFFFFFF

# --------------------------------------------------------------------------
# Precomputed row tables
# --------------------------------------------------------------------------
# For every possible 16-bit row we precompute:
#   * the row after sliding left / right
#   * the merge score produced by that slide
# The slide results are stored pre-shifted into each of the four row
# positions, so building a moved board needs no shifting of the result.

_ROW_LEFT = array("H", bytes(2 * 65536))      # row -> row slid left
_ROW_RIGHT = array("H", bytes(2 * 65536))     # row -> row slid right
_SCORE_LEFT = array("I", bytes(4 * 65536))    # row -> merge score (slide left)
_SCORE_RIGHT = array("I", bytes(4 * 65536))

# Pre-shifted 64-bit results: _L[r][row] already sits at row position r.
_L: list[array] = [array("Q", bytes(8 * 65536)) for _ in range(4)]
_R: list[array] = [array("Q", bytes(8 * 65536)) for _ in range(4)]

# Vertical moves. _U[c][row] takes a row *of the transposed board* (i.e.
# column c of the real board), slides it, and writes the result straight back
# into column c of a normal board. Folding the un-transpose into the table
# means a vertical move costs one transpose instead of two -- measured at
# 3.62us -> 2.29us, so it earns its 4 MB.
_U: list[array] = [array("Q", bytes(8 * 65536)) for _ in range(4)]
_D: list[array] = [array("Q", bytes(8 * 65536)) for _ in range(4)]

# row -> tuple of absolute cell indices that are empty, one table per row
# position. Replaces a 16-iteration Python loop in random_spawn
# (2.82us -> 1.61us) for ~11 MB.
_EC: list[list] = [[()] * 65536 for _ in range(4)]

# row -> number of empty cells in that row
_EMPTY = array("B", bytes(65536))
# row -> highest nibble value present in that row
_MAXEXP = array("B", bytes(65536))


def _slide_line_left(line):
    """Slide a 4-cell line (list of exponents) left. Returns (new_line, score).

    This is the single source of truth for merge semantics and is deliberately
    written in the most obvious way possible -- it runs 65536 times at import
    and never again.

    A tile that has just been formed by a merge cannot merge again during the
    same move: after consuming ``line[i]`` and ``line[j]`` we resume scanning
    from ``j + 1``, so the freshly written tile is never revisited.
    """
    out = []
    score = 0
    i = 0
    while i < 4:
        if line[i] == 0:
            i += 1
            continue
        # find the next non-empty cell
        j = i + 1
        while j < 4 and line[j] == 0:
            j += 1
        if j < 4 and line[j] == line[i] and line[i] < 15:
            merged = line[i] + 1
            out.append(merged)
            score += 1 << merged
            i = j + 1          # skip past the consumed partner: no double merge
        else:
            out.append(line[i])
            i = j
    out.extend([0] * (4 - len(out)))
    return out, score


def _build_tables():
    for row in range(65536):
        line = [(row >> (4 * c)) & 0xF for c in range(4)]

        empties = 0
        mx = 0
        for v in line:
            if v == 0:
                empties += 1
            elif v > mx:
                mx = v
        _EMPTY[row] = empties
        _MAXEXP[row] = mx

        left, lscore = _slide_line_left(line)
        lrow = left[0] | (left[1] << 4) | (left[2] << 8) | (left[3] << 12)
        _ROW_LEFT[row] = lrow
        _SCORE_LEFT[row] = lscore

        # sliding right == reverse, slide left, reverse
        rev = line[::-1]
        rout, rscore = _slide_line_left(rev)
        rout = rout[::-1]
        rrow = rout[0] | (rout[1] << 4) | (rout[2] << 8) | (rout[3] << 12)
        _ROW_RIGHT[row] = rrow
        _SCORE_RIGHT[row] = rscore

        empty_cols = tuple(c for c in range(4) if line[c] == 0)

        for r in range(4):
            _L[r][row] = lrow << (16 * r)
            _R[r][row] = rrow << (16 * r)
            _EC[r][row] = tuple(4 * r + c for c in empty_cols)
            # Scatter nibble k of the slid row into cell 4*k + r, i.e. bits
            # 16*k + 4*r -- column r of the output board.
            vu = vd = 0
            for k in range(4):
                vu |= left[k] << (16 * k + 4 * r)
                vd |= rout[k] << (16 * k + 4 * r)
            _U[r][row] = vu
            _D[r][row] = vd


_build_tables()

# Bind tables to locals at module scope; attribute lookup is expensive in the
# hot loop, so the move functions capture them via default arguments below.
_L0, _L1, _L2, _L3 = _L
_R0, _R1, _R2, _R3 = _R
_U0, _U1, _U2, _U3 = _U
_D0, _D1, _D2, _D3 = _D
_E0, _E1, _E2, _E3 = _EC


# --------------------------------------------------------------------------
# Transpose
# --------------------------------------------------------------------------
def transpose(b: int) -> int:
    """Transpose the 4x4 nibble matrix (swap rows and columns)."""
    a1 = b & 0xF0F00F0FF0F00F0F
    a2 = b & 0x0000F0F00000F0F0
    a3 = b & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


# --------------------------------------------------------------------------
# Moves
# --------------------------------------------------------------------------
def move(b: int, action: int):
    """Apply ``action`` to board ``b``.

    Returns ``(new_board, score_gained, moved)``. ``moved`` is False when the
    action is illegal (the board did not change); callers must not spawn a
    tile in that case.
    """
    if action == LEFT:
        r0 = b & 0xFFFF
        r1 = (b >> 16) & 0xFFFF
        r2 = (b >> 32) & 0xFFFF
        r3 = b >> 48
        nb = _L0[r0] | _L1[r1] | _L2[r2] | _L3[r3]
        sc = _SCORE_LEFT[r0] + _SCORE_LEFT[r1] + _SCORE_LEFT[r2] + _SCORE_LEFT[r3]
    elif action == RIGHT:
        r0 = b & 0xFFFF
        r1 = (b >> 16) & 0xFFFF
        r2 = (b >> 32) & 0xFFFF
        r3 = b >> 48
        nb = _R0[r0] | _R1[r1] | _R2[r2] | _R3[r3]
        sc = _SCORE_RIGHT[r0] + _SCORE_RIGHT[r1] + _SCORE_RIGHT[r2] + _SCORE_RIGHT[r3]
    elif action == UP:
        t = transpose(b)
        r0 = t & 0xFFFF
        r1 = (t >> 16) & 0xFFFF
        r2 = (t >> 32) & 0xFFFF
        r3 = t >> 48
        nb = _U0[r0] | _U1[r1] | _U2[r2] | _U3[r3]
        sc = _SCORE_LEFT[r0] + _SCORE_LEFT[r1] + _SCORE_LEFT[r2] + _SCORE_LEFT[r3]
    else:  # DOWN
        t = transpose(b)
        r0 = t & 0xFFFF
        r1 = (t >> 16) & 0xFFFF
        r2 = (t >> 32) & 0xFFFF
        r3 = t >> 48
        nb = _D0[r0] | _D1[r1] | _D2[r2] | _D3[r3]
        sc = _SCORE_RIGHT[r0] + _SCORE_RIGHT[r1] + _SCORE_RIGHT[r2] + _SCORE_RIGHT[r3]
    return nb, sc, nb != b


def move_no_score(b: int, action: int) -> int:
    """Apply ``action`` and return only the new board (== b when illegal).

    Used by search, which cares about the resulting position but not about the
    score increment; skipping the score table saves four lookups per node.
    """
    if action == LEFT:
        return (_L0[b & 0xFFFF] | _L1[(b >> 16) & 0xFFFF]
                | _L2[(b >> 32) & 0xFFFF] | _L3[b >> 48])
    if action == RIGHT:
        return (_R0[b & 0xFFFF] | _R1[(b >> 16) & 0xFFFF]
                | _R2[(b >> 32) & 0xFFFF] | _R3[b >> 48])
    t = transpose(b)
    if action == UP:
        return (_U0[t & 0xFFFF] | _U1[(t >> 16) & 0xFFFF]
                | _U2[(t >> 32) & 0xFFFF] | _U3[t >> 48])
    return (_D0[t & 0xFFFF] | _D1[(t >> 16) & 0xFFFF]
            | _D2[(t >> 32) & 0xFFFF] | _D3[t >> 48])


# --------------------------------------------------------------------------
# Board queries
# --------------------------------------------------------------------------
def empty_count(b: int) -> int:
    return (_EMPTY[b & 0xFFFF] + _EMPTY[(b >> 16) & 0xFFFF]
            + _EMPTY[(b >> 32) & 0xFFFF] + _EMPTY[b >> 48])


def max_tile_exp(b: int) -> int:
    """Largest tile on the board, as an exponent (0 for an empty board)."""
    m = _MAXEXP[b & 0xFFFF]
    v = _MAXEXP[(b >> 16) & 0xFFFF]
    if v > m:
        m = v
    v = _MAXEXP[(b >> 32) & 0xFFFF]
    if v > m:
        m = v
    v = _MAXEXP[b >> 48]
    if v > m:
        m = v
    return m


def max_tile(b: int) -> int:
    """Largest tile on the board as a face value (0 for an empty board)."""
    e = max_tile_exp(b)
    return (1 << e) if e else 0


def legal_actions(b: int):
    """Tuple of actions that would change the board.

    A direction is legal iff at least one line changes when slid that way, so
    this only needs row-table comparisons and a single transpose -- much
    cheaper than actually building all four boards.
    """
    out = []
    r0 = b & 0xFFFF
    r1 = (b >> 16) & 0xFFFF
    r2 = (b >> 32) & 0xFFFF
    r3 = b >> 48
    t = transpose(b)
    c0 = t & 0xFFFF
    c1 = (t >> 16) & 0xFFFF
    c2 = (t >> 32) & 0xFFFF
    c3 = t >> 48
    if _ROW_LEFT[c0] != c0 or _ROW_LEFT[c1] != c1 \
            or _ROW_LEFT[c2] != c2 or _ROW_LEFT[c3] != c3:
        out.append(UP)
    if _ROW_RIGHT[r0] != r0 or _ROW_RIGHT[r1] != r1 \
            or _ROW_RIGHT[r2] != r2 or _ROW_RIGHT[r3] != r3:
        out.append(RIGHT)
    if _ROW_RIGHT[c0] != c0 or _ROW_RIGHT[c1] != c1 \
            or _ROW_RIGHT[c2] != c2 or _ROW_RIGHT[c3] != c3:
        out.append(DOWN)
    if _ROW_LEFT[r0] != r0 or _ROW_LEFT[r1] != r1 \
            or _ROW_LEFT[r2] != r2 or _ROW_LEFT[r3] != r3:
        out.append(LEFT)
    return tuple(out)


def is_game_over(b: int) -> bool:
    """True when no direction changes the board."""
    r0 = b & 0xFFFF
    r1 = (b >> 16) & 0xFFFF
    r2 = (b >> 32) & 0xFFFF
    r3 = b >> 48
    if _EMPTY[r0] or _EMPTY[r1] or _EMPTY[r2] or _EMPTY[r3]:
        return False
    # The board is full, so a legal move requires an equal adjacent pair;
    # "row unchanged by a left slide" detects exactly that.
    if _ROW_LEFT[r0] != r0 or _ROW_LEFT[r1] != r1 \
            or _ROW_LEFT[r2] != r2 or _ROW_LEFT[r3] != r3:
        return False
    t = transpose(b)
    c0 = t & 0xFFFF
    c1 = (t >> 16) & 0xFFFF
    c2 = (t >> 32) & 0xFFFF
    c3 = t >> 48
    return (_ROW_LEFT[c0] == c0 and _ROW_LEFT[c1] == c1
            and _ROW_LEFT[c2] == c2 and _ROW_LEFT[c3] == c3)


# --------------------------------------------------------------------------
# Tile spawning
# --------------------------------------------------------------------------
def spawn_positions(b: int):
    """Indices of empty cells, low cell index first."""
    return list(_E0[b & 0xFFFF] + _E1[(b >> 16) & 0xFFFF]
                + _E2[(b >> 32) & 0xFFFF] + _E3[b >> 48])


def spawn_tile(b: int, cell: int, exp: int) -> int:
    """Place tile ``exp`` (1 for a 2, 2 for a 4) into empty ``cell``."""
    return b | (exp << (4 * cell))


def random_spawn(b: int, rng: Random) -> int:
    """Spawn one random tile: 90% a 2, 10% a 4, uniform over empty cells.

    Draws exactly one random number per spawn, which keeps seeded runs
    reproducible and keeps the hot loop cheap.
    """
    cells = (_E0[b & 0xFFFF] + _E1[(b >> 16) & 0xFFFF]
             + _E2[(b >> 32) & 0xFFFF] + _E3[b >> 48])
    n = len(cells)
    if n == 0:
        return b
    # One draw encodes both the cell choice and the 2-vs-4 choice.
    k, tens = divmod(rng.randrange(n * 10), 10)
    exp = 2 if tens == 0 else 1     # 1-in-10 chance of a 4
    return b | (exp << (4 * cells[k]))


def new_game(rng: Random) -> int:
    """A fresh board with the customary two starting tiles."""
    return random_spawn(random_spawn(0, rng), rng)


# --------------------------------------------------------------------------
# Conversion / display helpers (not used on the training hot path)
# --------------------------------------------------------------------------
def to_list(b: int):
    """16 tile face values, row-major (0 for empty)."""
    out = []
    for i in range(16):
        e = (b >> (4 * i)) & 0xF
        out.append((1 << e) if e else 0)
    return out


def board_to_rows(b: int):
    """4 rows of 4 tile face values."""
    flat = to_list(b)
    return [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]


def from_list(values) -> int:
    """Inverse of :func:`to_list`: accepts face values (0, 2, 4, 8, ...)."""
    b = 0
    for i, v in enumerate(values):
        if v:
            e = v.bit_length() - 1
            if (1 << e) != v:
                raise ValueError(f"{v} is not a power of two")
            if e > 15:
                raise ValueError(f"tile {v} does not fit in a nibble")
            b |= e << (4 * i)
    return b


def render(b: int) -> str:
    """Plain-text board, for tests and the CLI."""
    rows = board_to_rows(b)
    return "\n".join(
        " ".join(f"{v:5d}" if v else "    ." for v in row) for row in rows
    )
