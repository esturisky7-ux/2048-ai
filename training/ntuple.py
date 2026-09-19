"""N-tuple value network: the representation the learning agent uses.

Why an n-tuple network rather than a neural net
-----------------------------------------------
2048 has a small, discrete, highly local structure. An n-tuple network is a
linear model over an enormous sparse one-hot feature space: a handful of cell
groups ("tuples"), each looked up in its own table. Evaluating a board is a
few dozen array reads, and learning is a few dozen additions -- no matrix
multiplies, no autograd, no BLAS. On a dual-core 1.8 GHz CPU with no GPU that
is roughly two orders of magnitude cheaper per update than even a small MLP,
and it is also what the strongest published 2048 agents actually use
(Szubert & Jaskowski 2014; Wu et al. 2014).

Layout
------
A tuple is a set of board cells. For a 6-cell tuple the lookup index packs six
4-bit exponents into 24 bits, so the table has 16**6 = 16.7M entries.

Every tuple is shared across the 8 symmetries of the square (4 rotations x 2
reflections). The board has no intrinsic orientation, so a pattern learned in
one corner should transfer to the others; sharing weights this way multiplies
the effective training data by eight.

Speed
-----
Two tricks, both measured rather than assumed:

1. *Row-partial index tables.* An index is assembled from the four 16-bit
   board rows via precomputed lookup tables, so building it costs one array
   read per row the tuple touches instead of one shift-and-mask per cell.
2. *Generated straight-line code.* ``value`` and ``update`` are emitted as
   fully unrolled Python source with every table bound as a local. The
   equivalent interpreted loop over 32 variants is far slower, and the
   generated source is kept on the instance (``.source``) so it stays
   inspectable.

Storage
-------
Weights live in a memory-mapped file of float32. That gives three things at
once: processes share one copy of the table (``--workers 2`` needs no IPC for
weights), checkpointing only has to flush dirty pages, and a crash leaves
almost-current weights already on disk.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
from array import array
from pathlib import Path
from typing import Sequence

# Index tables take a few seconds to build and are identical for a given tuple
# set, so they are cached here. Every process that touches the network (two
# training workers, the evaluator, the dashboard) pays ~0.2 s instead of ~4 s.
# The cache is pure derived data: deleting it only costs one rebuild.
CACHE_DIR = Path(os.environ.get("AI2048_HOME")
                 or Path(__file__).resolve().parent.parent) / "data" / "tables"

# ---------------------------------------------------------------------------
# Tuple sets
# ---------------------------------------------------------------------------
# Cells are numbered row-major:
#     0  1  2  3
#     4  5  6  7
#     8  9 10 11
#    12 13 14 15

TUPLE_SETS: dict[str, tuple[tuple[int, ...], ...]] = {
    # The standard strong set: two horizontal 6-cell strips and two 2x3
    # blocks. 4 x 16**6 = 67M weights = 268 MB of float32.
    "4x6": (
        (0, 1, 2, 3, 4, 5),
        (4, 5, 6, 7, 8, 9),
        (0, 1, 2, 4, 5, 6),
        (4, 5, 6, 8, 9, 10),
    ),
    # Lighter: 4 x 16**5 = 4.2M weights = 17 MB. Trains faster per game in
    # wall-clock terms only marginally, but uses far less RAM and disk.
    "4x5": (
        (0, 1, 2, 3, 4),
        (4, 5, 6, 7, 8),
        (0, 1, 2, 4, 5),
        (4, 5, 6, 8, 9),
    ),
    # Rows plus overlapping squares; 8 x 16**4 = 524K weights = 2 MB.
    # Small enough to experiment with quickly, clearly weaker at the top end.
    "8x4": (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 4, 5),
        (1, 2, 5, 6),
        (2, 3, 6, 7),
        (4, 5, 8, 9),
        (5, 6, 9, 10),
        (6, 7, 10, 11),
    ),
}

DEFAULT_TUPLE_SET = "4x6"


def symmetry_permutations() -> list[list[int]]:
    """The 8 symmetries of the square, as permutations of cell indices."""
    perms = []
    for reflect in (False, True):
        for rot in range(4):
            p = []
            for i in range(16):
                r, c = divmod(i, 4)
                if reflect:
                    c = 3 - c
                for _ in range(rot):
                    r, c = c, 3 - r          # rotate 90 degrees
                p.append(4 * r + c)
            perms.append(p)
    assert len({tuple(p) for p in perms}) == 8, "symmetry group is wrong"
    return perms


class NTupleNetwork:
    """A weight table plus generated ``value``/``update`` functions.

    Parameters
    ----------
    tuple_set:
        Key into :data:`TUPLE_SETS`, or an explicit sequence of cell tuples.
    path:
        File backing the weights. ``None`` keeps them in anonymous memory
        (useful for tests); otherwise the file is created sparse if missing
        and memory-mapped shared, so several processes train on one copy.
    readonly:
        Map the file read-only (for evaluation and the dashboard, so watching
        a game can never corrupt a training run).
    """

    def __init__(self, tuple_set: str | Sequence = DEFAULT_TUPLE_SET,
                 path: str | None = None, readonly: bool = False):
        if isinstance(tuple_set, str):
            if tuple_set not in TUPLE_SETS:
                raise ValueError(
                    f"unknown tuple set {tuple_set!r}; "
                    f"choose from {sorted(TUPLE_SETS)}")
            self.tuple_set_name = tuple_set
            tuples = TUPLE_SETS[tuple_set]
        else:
            self.tuple_set_name = "custom"
            tuples = tuple(tuple(t) for t in tuple_set)
        self.tuples = tuples
        self.path = path
        self.readonly = readonly

        self.sizes = [1 << (4 * len(t)) for t in tuples]
        self.offsets = []
        off = 0
        for s in self.sizes:
            self.offsets.append(off)
            off += s
        self.n_weights = off
        self.n_variants = len(tuples) * 8

        self._open_storage()
        self._build_index_tables()
        self._generate_code()

    # -- storage -----------------------------------------------------------
    def _open_storage(self) -> None:
        nbytes = 4 * self.n_weights
        if self.path is None:
            self._file = None
            self._mm = mmap.mmap(-1, nbytes)
        else:
            new = (not os.path.exists(self.path)
                   or os.path.getsize(self.path) != nbytes)
            if new:
                if self.readonly:
                    raise FileNotFoundError(
                        f"weight file {self.path} missing or wrong size "
                        f"(expected {nbytes} bytes)")
                Path(self.path).resolve().parent.mkdir(parents=True,
                                                       exist_ok=True)
                # Sparse allocation: instant, and pages materialise as zeros
                # only when actually touched.
                with open(self.path, "wb") as f:
                    f.truncate(nbytes)
            mode = "rb" if self.readonly else "r+b"
            self._file = open(self.path, mode)
            if self.readonly:
                self._mm = mmap.mmap(self._file.fileno(), 0,
                                     access=mmap.ACCESS_READ)
            else:
                self._mm = mmap.mmap(self._file.fileno(), 0,
                                     access=mmap.ACCESS_WRITE)
        self.weights = memoryview(self._mm).cast("f")

    def flush(self) -> None:
        """Push dirty weight pages to disk. Cheap: only touched pages move."""
        if self._file is not None and not self.readonly:
            self._mm.flush()

    def close(self) -> None:
        try:
            self.weights.release()
        except Exception:
            pass
        self._mm.close()
        if self._file is not None:
            self._file.close()

    def nbytes(self) -> int:
        return 4 * self.n_weights

    # -- index tables ------------------------------------------------------
    def _build_index_tables(self) -> None:
        """For each (tuple, symmetry) variant, tables turning rows into indices.

        ``self.variants[v]`` is a list of ``(row_position, table)`` pairs; the
        variant's lookup index is the bitwise OR of ``table[row_value]`` over
        those pairs, plus the tuple's base offset.
        """
        if self._load_index_cache():
            return
        perms = symmetry_permutations()
        self.variants = []
        self.variant_tuple = []
        for ti, cells in enumerate(self.tuples):
            base = self.offsets[ti]
            for perm in perms:
                mapped = [perm[c] for c in cells]
                by_row: dict[int, list[tuple[int, int]]] = {}
                for k, cell in enumerate(mapped):
                    row, col = divmod(cell, 4)
                    by_row.setdefault(row, []).append((k, col))
                parts = []
                for row in sorted(by_row):
                    tbl = array("i", bytes(4 * 65536))
                    for rv in range(65536):
                        acc = 0
                        for k, col in by_row[row]:
                            acc |= ((rv >> (4 * col)) & 0xF) << (4 * k)
                        tbl[rv] = acc
                    parts.append((row, tbl))
                # Fold the tuple's base offset into the first part's table so
                # the generated code needs no extra addition.
                if base:
                    first = parts[0][1]
                    for rv in range(65536):
                        first[rv] += base
                self.variants.append(parts)
                self.variant_tuple.append(ti)
        self._save_index_cache()

    # -- index table cache -------------------------------------------------
    def _cache_path(self) -> Path:
        key = hashlib.sha1(
            json.dumps([list(t) for t in self.tuples]).encode()).hexdigest()[:16]
        return Path(CACHE_DIR) / f"idx-{self.tuple_set_name}-{key}.bin"

    def _load_index_cache(self) -> bool:
        path = self._cache_path()
        try:
            with open(path, "rb") as f:
                hdr_len = int.from_bytes(f.read(4), "little")
                layout = json.loads(f.read(hdr_len).decode())
                self.variants = []
                self.variant_tuple = layout["variant_tuple"]
                for rows in layout["rows"]:
                    parts = []
                    for row in rows:
                        tbl = array("i")
                        tbl.frombytes(f.read(4 * 65536))
                        if len(tbl) != 65536:
                            raise ValueError("truncated index cache")
                        parts.append((row, tbl))
                    self.variants.append(parts)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False
        return True

    def _save_index_cache(self) -> None:
        path = self._cache_path()
        try:
            Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
            layout = {
                "variant_tuple": list(self.variant_tuple),
                "rows": [[row for row, _ in parts] for parts in self.variants],
            }
            hdr = json.dumps(layout).encode()
            tmp = path.with_suffix(".bin.tmp")
            with open(tmp, "wb") as f:
                f.write(len(hdr).to_bytes(4, "little"))
                f.write(hdr)
                for parts in self.variants:
                    for _, tbl in parts:
                        f.write(tbl.tobytes())
            os.replace(tmp, path)
        except OSError:
            pass          # a cache miss is never fatal

    # -- code generation ---------------------------------------------------
    def _generate_code(self) -> None:
        ns: dict = {"W": self.weights}
        head = [
            "def value(b):",
            "    r0 = b & 0xFFFF",
            "    r1 = (b >> 16) & 0xFFFF",
            "    r2 = (b >> 32) & 0xFFFF",
            "    r3 = b >> 48",
            "    return (",
        ]
        terms = []
        for v, parts in enumerate(self.variants):
            expr = " | ".join(f"T{v}_{row}[r{row}]" for row, _ in parts)
            for row, tbl in parts:
                ns[f"T{v}_{row}"] = tbl
            terms.append(f"        W[{expr}]")
        head.append(" +\n".join(terms))
        head.append("    )")
        value_src = "\n".join(head)

        upd = [
            "def update(b, d):",
            "    r0 = b & 0xFFFF",
            "    r1 = (b >> 16) & 0xFFFF",
            "    r2 = (b >> 32) & 0xFFFF",
            "    r3 = b >> 48",
        ]
        for v, parts in enumerate(self.variants):
            expr = " | ".join(f"T{v}_{row}[r{row}]" for row, _ in parts)
            upd.append(f"    i = {expr}")
            upd.append("    W[i] = W[i] + d")
        update_src = "\n".join(upd)

        # value_and_update: the TD loop always evaluates the afterstate it is
        # about to correct, so fusing the two halves the index arithmetic.
        vu = [
            "def update_scaled(b, err, alpha):",
            "    r0 = b & 0xFFFF",
            "    r1 = (b >> 16) & 0xFFFF",
            "    r2 = (b >> 32) & 0xFFFF",
            "    r3 = b >> 48",
            "    d = alpha * err",
        ]
        for v, parts in enumerate(self.variants):
            expr = " | ".join(f"T{v}_{row}[r{row}]" for row, _ in parts)
            vu.append(f"    i = {expr}")
            vu.append("    W[i] = W[i] + d")
        update_scaled_src = "\n".join(vu)

        self.source = value_src + "\n\n" + update_src + "\n\n" + update_scaled_src
        exec(compile(self.source, "<ntuple-generated>", "exec"), ns)
        self.value = ns["value"]
        self.update = ns["update"]
        self.update_scaled = ns["update_scaled"]
        self._ns = ns

    # -- reference implementation (for tests) ------------------------------
    def indices(self, b: int) -> list[int]:
        """Weight indices touched by board ``b``, computed the slow, obvious way.

        Used only by the test suite to verify the generated code.
        """
        perms = symmetry_permutations()
        out = []
        for ti, cells in enumerate(self.tuples):
            for perm in perms:
                idx = 0
                for k, c in enumerate(cells):
                    cell = perm[c]
                    nib = (b >> (4 * cell)) & 0xF
                    idx |= nib << (4 * k)
                out.append(self.offsets[ti] + idx)
        return out

    def value_reference(self, b: int) -> float:
        W = self.weights
        return sum(W[i] for i in self.indices(b))

    # -- housekeeping ------------------------------------------------------
    def nonzero_count(self, sample: int = 0) -> int:
        """Number of non-zero weights (or an estimate from a sample)."""
        W = self.weights
        n = self.n_weights
        if sample and sample < n:
            step = n // sample
            hits = sum(1 for i in range(0, n, step) if W[i])
            return int(hits * step)
        return sum(1 for i in range(n) if W[i])

    def describe(self) -> dict:
        return {
            "tuple_set": self.tuple_set_name,
            "tuples": [list(t) for t in self.tuples],
            "n_weights": self.n_weights,
            "n_variants": self.n_variants,
            "bytes": self.nbytes(),
        }

    def __repr__(self) -> str:
        return (f"<NTupleNetwork {self.tuple_set_name} "
                f"{len(self.tuples)} tuples x 8 symmetries, "
                f"{self.n_weights:,} weights, {self.nbytes()/1e6:.0f} MB>")
