"""tools/make_profile.py - the expert-cache profile (`data/expert-profile.bin`): every (layer, expert) pair, ranked.

`--expert-cache auto` fills the VRAM slots it can afford with the profile's pairs in order, so the ranking decides
which experts start resident (the adaptive tier then swaps in what a conversation routes most).  A profile that
ranks fewer pairs than a card can hold caps the cache (issue #46: a 32 GB card stopped at 8,000 slots); this tool
writes one that ranks all 24,576.

The order: the base profile's ranking (default: the shipped data/expert-profile.bin), then the pairs your routing
traces used, most frequent first, then every pair still missing, interleaved across the layers.  The base is only
*appended to*, never reordered - and the shipped profile already ranks all 24,576 pairs, so with it a trace is a
no-op.  `--reorder` ranks the traces first and lets the base fill the rest, which is how a workload's trace decides
the top; `--no-base` drops the base entirely.

WATCH THE BASE: `take()` skips a pair it has already ranked, and the shipped base ranks all 24,576, so a plain
`--base` run cannot be moved by any trace - the output is the base again.  `--reorder` (traces first) or
`--no-base` is what makes a trace decide the top.  Which ordering is in use is not cosmetic: the layer-split cost
model reads this RANKING as if it were a frequency curve, and the coverage curve in src/program/generate.cpp
carries the measured hit rates that say how far off that goes (94.9% claimed against 69.2% measured at 5,805 pairs
held, on the IQ3_S 4-way rig).

    python tools/make_profile.py [TRACE ...] [--base data/expert-profile.bin | --no-base] [--reorder] [--out PATH]
                                 [--n-expert 256]      (a pruned model: GSQ-RCO Coder keeps 256 of 512)

A routing trace comes from a one-shot engine run with `--dump-routing FILE` (a prompt typical of your use; the
routed experts of every layer and position are written).  Point the model config's `--expert-profile` at the result.
Both this tool and `bench_eviction.py` refuse partial headers/bodies, invalid expert counts and invalid
weights before returning a result. Zero placeholder weights remain valid. Framing cannot detect missing
whole records or establish request/phase coverage; those require the capture's independent run binding.
"""
import argparse
import math
import struct
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_LAYER, N_EXPERT = 48, 512
MAGIC, VERSION = b"STRP", 1


def read_profile(path, n_expert=N_EXPERT):
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit(f"{path}: not a Strata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if (nl, ne) != (N_LAYER, n_expert):
        raise SystemExit(f"{path}: {nl}x{ne}, not {N_LAYER}x{n_expert}")
    return [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]


def trace_records(path):
    """Read complete records; never interpret a malformed capture's valid prefix as evidence."""
    with Path(path).open("rb") as source:
        offset = 0
        while header := source.read(8):
            if len(header) != 8:
                raise ValueError(f"{path}: truncated routing header at byte {offset}")
            layer, k = struct.unpack("<ii", header)
            if not 0 < k <= 4096:
                raise ValueError(f"{path}: implausible expert count {k} at byte {offset}")
            body = source.read(8 * k)
            if len(body) != 8 * k:
                raise ValueError(f"{path}: truncated routing body at byte {offset + 8}")
            experts = struct.unpack_from(f"<{k}i", body)
            weights = struct.unpack_from(f"<{k}f", body, 4 * k)
            if any(not math.isfinite(weight) or weight < 0 for weight in weights):
                raise ValueError(f"{path}: invalid routing weights at byte {offset}")
            yield layer, experts
            offset += 8 + 8 * k


def read_trace(path, n_expert=N_EXPERT):
    """Count in-range pairs only after checking every record's complete IDs and weights."""
    freq = defaultdict(int)
    for layer, experts in trace_records(path):
        for e in experts:
            if 0 <= layer < N_LAYER and 0 <= e < n_expert:
                freq[(layer, e)] += 1
    return freq


def write_profile(path, ranked, n_expert=N_EXPERT):
    table = [[-1] * n_expert for _ in range(N_LAYER)]
    for slot, (layer, e) in enumerate(ranked):
        table[layer][e] = slot
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, N_LAYER, n_expert, len(ranked), len(ranked)))
        for layer, e in ranked:
            f.write(struct.pack("<HH", layer, e))
        for layer in range(N_LAYER):
            f.write(struct.pack("<%di" % n_expert, *table[layer]))


def rank_profile(base_pairs, trace_freq, n_expert=N_EXPERT, no_base=False, reorder=False):
    """The ranked (layer, expert) list and the counts it was built from (`base`, `trace`, `fill`).

    Default: the base's order, then the traces' pairs most frequent first, then the fill.  With `reorder` (or
    `no_base`) the traces' pairs come first, so a base that already ranks every pair no longer hides them; the base
    (unless `no_base`) and then the fill follow.  A base is never reordered against itself - it only ever fills."""
    ranked, seen, counts = [], set(), defaultdict(int)

    def take(pairs, key):
        for p in pairs:
            p = (int(p[0]), int(p[1]))
            if p not in seen:
                seen.add(p)
                ranked.append(p)
                counts[key] += 1

    trace_pairs = [p for p, _ in sorted(trace_freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    if no_base or reorder:
        take(trace_pairs, "trace")
    if not no_base:
        take(base_pairs, "base")
    if not (no_base or reorder):
        take(trace_pairs, "trace")
    take(((layer, e) for e in range(n_expert) for layer in range(N_LAYER)), "fill")
    return ranked, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="*", help="routing traces from --dump-routing")
    ap.add_argument("--base", default=str(ROOT / "data" / "expert-profile.bin"), help="ranking to keep first")
    ap.add_argument("--no-base", action="store_true", help="rank by the traces only")
    ap.add_argument("--reorder", action="store_true",
                    help="rank the traces' pairs before the base's (the base, then the fill, follow); needed with a "
                         "base that already ranks every pair, where --base alone cannot change the order")
    ap.add_argument("--out", default=str(ROOT / "data" / "expert-profile.bin"))
    ap.add_argument("--n-expert", type=int, default=N_EXPERT, help="experts per layer (default 512)")
    a = ap.parse_args()

    ne = a.n_expert
    freq = defaultdict(int)
    for t in a.traces:
        for p, c in read_trace(t, ne).items():
            freq[p] += c
    base_pairs = [] if a.no_base else read_profile(a.base, ne)
    ranked, counts = rank_profile(base_pairs, freq, ne, no_base=a.no_base, reorder=a.reorder)
    write_profile(a.out, ranked, ne)
    assert read_profile(a.out, ne) == ranked, "the profile did not survive the round trip"
    print(f"wrote {a.out}: {len(ranked)} ranked pairs ({counts['base']} from the base, {counts['trace']} from the "
          f"traces, {counts['fill']} filled in)")


if __name__ == "__main__":
    main()
