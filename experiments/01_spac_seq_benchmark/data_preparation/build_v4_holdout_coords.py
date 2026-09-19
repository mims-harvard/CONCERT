#!/usr/bin/env python
"""Bridge benchmark_v4 (SPAC-seq) -> CONCERT coordinate-keyed holdout files.

Steps
  1. Reproduce v3: run the unchanged build_v3_holdout_coords.build() into a scratch path and assert
     it equals v3_holdout_coords/holdout_v3_spac_{seen,unseen}.json exactly. This validates the
     format and the translation before anything v4 is written.
  2. Diagnostic: also build the v3 unseen KO block from benchmark_v3/eval_panel.json units only
     (v3 build() lists every file in holdout_ko/, which may contain files from earlier builds) and
     report whether it differs from the shipped v3 file.
  3. Write v4: unassigned block from benchmark_v4/holdout_unassigned, KO block from the
     benchmark_v4/eval_panel.json units (never a directory listing), same JSON layout as v3;
     verify every coordinate against the raw files CONCERT trains on.
  4. Seen: the v4 seen file is a copy of the v3 seen file after asserting that a rebuild from
     benchmark_v4 is identical.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_v3_holdout_coords as b3  # noqa: E402

EVAL = "/n/holylfs06/LABS/mzitnik_lab/Lab/xianglin226/spac_seq_eval"
PREP = f"{EVAL}/multi_1-5/prepared"
SP = "/n/holylabs/LABS/mzitnik_lab/Users/xianglin226/spaVAE/datasets/newdata/spac_seq"
SLIDE_TO_SID = {"1-1": 0, "1-2": 1, "1-3": 2, "1-4": 3, "1-5": 4}
RAW = {s: os.path.join(SP, f"raw_output_{s}.h5ad") for s in SLIDE_TO_SID}


def build_from_bench(bench_dir, regime, ko_from_panel=True):
    """Same output layout as build_v3_holdout_coords.build, reading an arbitrary benchmark dir."""
    unassigned, ko, stats = {}, {}, []
    cache = {}

    def coords_of(sid_str):
        if sid_str not in cache:
            cache[sid_str] = b3.slide_coords(PREP, sid_str, RAW.get(sid_str))
        return cache[sid_str]

    hu_dir = os.path.join(bench_dir, "holdout_unassigned")
    for fn in sorted(os.listdir(hu_dir)):
        d = json.load(open(os.path.join(hu_dir, fn)))
        sid_str = d["slide_id"]
        if sid_str not in SLIDE_TO_SID:
            continue
        coords = coords_of(sid_str)
        idx = np.asarray(d["union_holdout_idx"], dtype=np.int64)
        assert ((idx >= 0) & (idx < len(coords))).all(), fn
        sel = np.unique(coords[idx], axis=0)
        unassigned[str(SLIDE_TO_SID[sid_str])] = sel.tolist()
        stats.append(f"  unassigned {sid_str}: {len(idx)} idx -> {len(sel)} uniq coords")

    if regime == "unseen":
        panel = json.load(open(os.path.join(bench_dir, "eval_panel.json")))
        if ko_from_panel:
            files = [f"{u['gene']}__target_slide_{u['target_slide']}.json" for u in panel["eval_units"]]
        else:
            files = sorted(os.listdir(os.path.join(bench_dir, "holdout_ko")))
        agg = {}
        for fn in files:
            d = json.load(open(os.path.join(bench_dir, "holdout_ko", fn)))
            sid_str = d["slide_id"]
            coords = coords_of(sid_str)
            idx = np.asarray(d["ko_holdout_idx"], dtype=np.int64)
            assert ((idx >= 0) & (idx < len(coords))).all(), fn
            agg.setdefault(sid_str, []).append(coords[idx])
        for sid_str in sorted(agg):
            arrs = agg[sid_str]
            sel = np.unique(np.concatenate(arrs, axis=0), axis=0)
            ko[str(SLIDE_TO_SID[sid_str])] = sel.tolist()
            stats.append(f"  ko {sid_str}: {sum(len(a) for a in arrs)} idx -> {len(sel)} uniq coords")
    out = {"dataset": os.path.basename(PREP), "regime": regime,
           "slide_to_sid": SLIDE_TO_SID, "unassigned": unassigned, "ko": ko}
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3_dir", default=f"{EVAL}/v3_holdout_coords")
    ap.add_argument("--outdir", default=f"{EVAL}/v4_holdout_coords")
    ap.add_argument("--scratch", required=True, help="scratch dir for the v3 reproduction")
    ap.add_argument("--bench_subdir", default="benchmark_v4")
    args = ap.parse_args()
    os.makedirs(args.scratch, exist_ok=True)
    assert os.path.abspath(args.outdir) != os.path.abspath(args.v3_dir)

    # 1. v3 reproduction with the unchanged v3 code
    for regime in ("seen", "unseen"):
        rep_path = os.path.join(args.scratch, f"repro_holdout_v3_spac_{regime}.json")
        b3.build(PREP, regime, SLIDE_TO_SID, rep_path, raw_files=RAW)
        shipped = json.load(open(os.path.join(args.v3_dir, f"holdout_v3_spac_{regime}.json")))
        repro = json.load(open(rep_path))
        assert repro == shipped, f"v3 {regime} reproduction differs from shipped file"
        print(f"[v3 repro] {regime}: IDENTICAL to shipped holdout_v3_spac_{regime}.json")

    # 2. diagnostic: v3 unseen KO block from panel units only
    v3_panel_only, _ = build_from_bench(os.path.join(PREP, "benchmark_v3"), "unseen", ko_from_panel=True)
    shipped_u = json.load(open(os.path.join(args.v3_dir, "holdout_v3_spac_unseen.json")))
    for sid in sorted(set(shipped_u["ko"]) | set(v3_panel_only["ko"])):
        a = len(shipped_u["ko"].get(sid, []))
        b = len(v3_panel_only["ko"].get(sid, []))
        print(f"[v3 diag] sid{sid}: shipped KO coords={a}  panel-units-only KO coords={b}  diff={a - b}")
    stale = sorted(set(os.listdir(os.path.join(PREP, "benchmark_v3", "holdout_ko")))
                   - {f"{u['gene']}__target_slide_{u['target_slide']}.json"
                      for u in json.load(open(os.path.join(PREP, "benchmark_v3", "eval_panel.json")))["eval_units"]})
    print(f"[v3 diag] holdout_ko files not in the v3 eval panel ({len(stale)}): {stale}")

    # 3. v4 unseen
    os.makedirs(args.outdir, exist_ok=True)
    bench = os.path.join(PREP, args.bench_subdir)
    v4u, stats = build_from_bench(bench, "unseen", ko_from_panel=True)
    ko_files = set(os.listdir(os.path.join(bench, "holdout_ko")))
    panel_files = {f"{u['gene']}__target_slide_{u['target_slide']}.json"
                   for u in json.load(open(os.path.join(bench, "eval_panel.json")))["eval_units"]}
    assert ko_files == panel_files, f"benchmark_v4 holdout_ko files != panel units: {ko_files ^ panel_files}"
    assert v4u["unassigned"] == shipped_u["unassigned"], "v4 unassigned block differs from v3"
    out_u = os.path.join(args.outdir, "holdout_v4_spac_unseen.json")
    b3.verify_against_raw(v4u, RAW, SLIDE_TO_SID)
    json.dump(v4u, open(out_u, "w"))
    print(f"[v4] wrote {out_u}")
    print("\n".join(stats))

    # 4. v4 seen = v3 seen (assert rebuild identical, then copy)
    v4s, _ = build_from_bench(bench, "seen")
    shipped_s = json.load(open(os.path.join(args.v3_dir, "holdout_v3_spac_seen.json")))
    assert v4s == shipped_s, "seen rebuild from benchmark_v4 differs from v3 seen file"
    out_s = os.path.join(args.outdir, "holdout_v4_spac_seen.json")
    shutil.copyfile(os.path.join(args.v3_dir, "holdout_v3_spac_seen.json"), out_s)
    print(f"[v4] seen identical to v3 -> copied to {out_s}")


if __name__ == "__main__":
    main()
