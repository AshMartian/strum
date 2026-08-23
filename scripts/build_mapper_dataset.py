"""build_mapper_dataset.py — Build (BP onset features, GT fret labels) dataset.

For each song with paired isolated guitar.ogg + notes.mid:
  1. Run Basic Pitch on the guitar stem (or load cached BP output)
  2. Group BP notes into onset clusters
  3. Parse GT MIDI (PART GUITAR Expert) to get true fret events
  4. Match each BP onset to nearest GT onset (±MATCH_TOL_MS)
  5. Extract feature vector + 5-binary fret label per matched onset
  6. Cache as one .npz per song under --cache-dir

Output npz fields:
  X        (N, F)  features per onset
  Y        (N, 5)  binary fret labels [G,R,Y,B,O]
  meta     dict-like header (song_id, n_onsets, etc.)

Usage:
  python scripts/build_mapper_dataset.py \
      --songs-root /mnt/ml-data/training_songs \
      --cache-dir /mnt/ml-data/fret_mapper_cache \
      --max-songs 200 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Quiet noisy libs before importing them
logging.getLogger("root").setLevel(logging.ERROR)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from src.catalog_task_manifest import resolve_catalog_task_manifest_songs  # noqa: E402
from src.preprocessing.parsers.guitar_parser import GuitarParser  # noqa: E402

log = logging.getLogger("build_mapper_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MATCH_TOL_MS = 50.0
ONSET_GROUP_SEC = 0.080
CONTEXT = 2  # include ±2 neighbour onsets in features


# ─────────────────────────── Feature extraction ─────────────────────────────
def featurize_onsets(onsets: list[dict]) -> np.ndarray:
    """Return (N, F) feature matrix.

    Per-onset features:
      - 12 floats: pitch-class histogram (max amp per PC, normalized)
      - 1 float: log(n_distinct_pitches)
      - 1 float: root pitch / 127
      - 1 float: pitch_span (max-min) / 24
      - 1 float: max_amp
      - 1 float: max_duration / 2.0
      - 1 float: 1 if "power chord" pattern (root+5th present)
      - 1 float: 1 if "chord" pattern (>=2 PCs, not power)
    Plus context: same 19 features for previous/next onset (zero-padded)
    Total: 19 * (1 + 2*CONTEXT) = 19 * 5 = 95 features
    """
    N = len(onsets)
    per = 19
    feats = np.zeros((N, per), dtype=np.float32)
    for i, ons in enumerate(onsets):
        pitches = ons["pitches"]
        amps = ons["amps"]
        durs = ons["durations"]
        if not pitches:
            continue
        pcs = np.array([p % 12 for p in pitches])
        max_per_pc = np.zeros(12, dtype=np.float32)
        for pc, a in zip(pcs, amps, strict=False):
            if a > max_per_pc[pc]:
                max_per_pc[pc] = a
        norm = max_per_pc.max()
        if norm > 0:
            max_per_pc = max_per_pc / norm
        feats[i, :12] = max_per_pc
        n_distinct_pitches = len(set(pitches))
        feats[i, 12] = np.log1p(n_distinct_pitches)
        root = min(pitches)
        feats[i, 13] = root / 127.0
        feats[i, 14] = (max(pitches) - min(pitches)) / 24.0
        feats[i, 15] = float(max(amps))
        feats[i, 16] = min(2.0, float(max(durs))) / 2.0
        pcset = set(pcs.tolist())
        root_pc = root % 12
        is_power = ((root_pc + 7) % 12) in pcset and len(pcset) <= 3 and len(pcset) >= 2
        is_chord = len(pcset) >= 2 and not is_power
        feats[i, 17] = float(is_power)
        feats[i, 18] = float(is_chord)

    # Concatenate context: prev-2, prev-1, self, next-1, next-2
    F = per * (1 + 2 * CONTEXT)
    out = np.zeros((N, F), dtype=np.float32)
    for i in range(N):
        col = 0
        for off in range(-CONTEXT, CONTEXT + 1):
            j = i + off
            if 0 <= j < N:
                out[i, col : col + per] = feats[j]
            col += per
    return out


def group_notes_by_onset(notes, window_sec: float = ONSET_GROUP_SEC):
    """Same as scripts/guitar_basicpitch.py but inlined to avoid heavy import."""
    notes = sorted(notes, key=lambda n: n[0])
    onsets: list[dict] = []
    cur_t = -1e9
    cur: dict | None = None
    for start, end, pitch, amp, _ in notes:
        if start - cur_t > window_sec:
            if cur is not None:
                onsets.append(cur)
            cur = {"time": float(start), "pitches": [], "durations": [], "amps": []}
            cur_t = float(start)
        cur["pitches"].append(int(pitch))
        cur["durations"].append(float(end - start))
        cur["amps"].append(float(amp))
    if cur is not None:
        onsets.append(cur)
    return onsets


# ─────────────────────────── GT chart parsing ───────────────────────────────
def parse_gt_chart(midi_path: Path, instrument: str = "guitar") -> list[dict]:
    """Return onset/fret labels from an Expert Guitar or Bass chart track."""
    chart = GuitarParser().parse(midi_path, instrument=instrument)
    if not chart.notes:
        return []
    by_tick: dict[int, dict] = {}
    for n in chart.notes:
        slot = by_tick.setdefault(n.tick, {"time_sec": n.time_ms / 1000.0, "frets": set()})
        slot["frets"].add(int(n.fret))
    return sorted(by_tick.values(), key=lambda x: x["time_sec"])


def match_bp_to_gt(
    bp_onsets: list[dict], gt_onsets: list[dict], tol_sec: float = MATCH_TOL_MS / 1000.0
):
    """Greedy nearest match. Returns list of (bp_idx, gt_idx)."""
    pairs = []
    j_start = 0
    for i, bp in enumerate(bp_onsets):
        t = bp["time"]
        best_j = -1
        best_dt = tol_sec
        for j in range(j_start, len(gt_onsets)):
            dt = gt_onsets[j]["time_sec"] - t
            if dt < -tol_sec:
                j_start = j + 1
                continue
            if dt > tol_sec:
                break
            if abs(dt) < best_dt:
                best_dt = abs(dt)
                best_j = j
        if best_j >= 0:
            pairs.append((i, best_j))
    return pairs


# ─────────────────────────── Per-song worker ────────────────────────────────
def process_song(args: tuple) -> dict:
    (
        song_id,
        audio_path,
        midi_path,
        cache_dir,
        onset_thr,
        frame_thr,
        min_note_len,
        instrument,
        split,
    ) = args
    audio_path = Path(audio_path)
    midi_path = Path(midi_path)
    cache_dir = Path(cache_dir)
    out_path = cache_dir / f"{song_id}.npz"
    if out_path.exists():
        return {"song": song_id, "status": "cached", "n": 0}
    if not audio_path.exists() or not midi_path.exists():
        return {"song": song_id, "status": "missing", "n": 0}
    try:
        # Lazy-import basic_pitch in worker
        import logging as _lg

        _lg.getLogger("root").setLevel(_lg.ERROR)
        from basic_pitch.inference import predict

        t0 = time.time()
        _, _, raw_notes = predict(
            str(audio_path),
            onset_threshold=onset_thr,
            frame_threshold=frame_thr,
            minimum_note_length=min_note_len,
        )
        bp_t = time.time() - t0

        bp_onsets = group_notes_by_onset(raw_notes)
        gt_onsets = parse_gt_chart(midi_path, instrument)
        if not bp_onsets or not gt_onsets:
            return {"song": song_id, "status": "empty", "n": 0}
        pairs = match_bp_to_gt(bp_onsets, gt_onsets)
        if not pairs:
            return {"song": song_id, "status": "no_match", "n": 0}

        feats = featurize_onsets(bp_onsets)
        bp_idx = np.array([p[0] for p in pairs])
        gt_idx = np.array([p[1] for p in pairs])
        X = feats[bp_idx]
        Y = np.zeros((len(pairs), 5), dtype=np.float32)
        for k, gj in enumerate(gt_idx):
            for f in gt_onsets[gj]["frets"]:
                if 0 <= f <= 4:
                    Y[k, f] = 1.0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            X=X,
            Y=Y,
            song_id=song_id,
            n_bp=len(bp_onsets),
            n_gt=len(gt_onsets),
            n_matched=len(pairs),
            bp_seconds=bp_t,
            # A catalog task view owns split assignment.  Persist it with the
            # derived examples so the trainer cannot re-shuffle songs and leak
            # samples across the OCTAVE-approved validation boundary.
            split=split,
        )
        return {"song": song_id, "status": "ok", "n": len(pairs), "bp_t": bp_t}
    except Exception as e:
        return {"song": song_id, "status": f"error: {e}", "n": 0, "trace": traceback.format_exc()}


def build_payloads(
    candidates: list[tuple[str, Path, Path, str]],
    *,
    cache_dir: Path,
    onset_threshold: float,
    frame_threshold: float,
    min_note_length: int,
    instrument: str,
) -> list[tuple[object, ...]]:
    """Build immutable per-song preprocessing inputs without dropping the split.

    The worker treats catalog train/validation membership as a hard boundary.
    Each cache file therefore receives its owning song split rather than
    deriving one later from an untrusted cache directory.
    """
    return [
        (
            song_id,
            str(audio_path),
            str(midi_path),
            str(cache_dir),
            onset_threshold,
            frame_threshold,
            min_note_length,
            instrument,
            split,
        )
        for song_id, audio_path, midi_path, split in candidates
    ]


# ─────────────────────────── Main ───────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs-root", type=Path, help="legacy raw-song directory tree")
    ap.add_argument("--catalog-manifest", type=Path, help="STRUM fret-mapper catalog task view")
    ap.add_argument(
        "--catalog-root", type=Path, help="OCTAVE catalog root for runtime revalidation"
    )
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--max-songs", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--onset-threshold", type=float, default=0.5)
    ap.add_argument("--frame-threshold", type=float, default=0.3)
    ap.add_argument("--min-note-length", type=int, default=11)
    args = ap.parse_args()

    using_catalog = args.catalog_manifest is not None or args.catalog_root is not None
    if using_catalog and (args.catalog_manifest is None or args.catalog_root is None):
        ap.error("--catalog-manifest and --catalog-root must be used together")
    if using_catalog and args.songs_root is not None:
        ap.error("choose either legacy --songs-root or a catalog task manifest")
    if not using_catalog and args.songs_root is None:
        ap.error("--songs-root is required without a catalog task manifest")

    if using_catalog:
        manifest = json.loads(args.catalog_manifest.read_text(encoding="utf-8"))
        task = manifest.get("task")
        if not isinstance(task, dict) or task.get("kind") not in {
            "fret_mapper_guitar",
            "fret_mapper_bass",
        }:
            ap.error("catalog manifest must use fret_mapper_guitar or fret_mapper_bass")
        instrument = task["instrument"]
        candidates = [
            (
                song["source_id"],
                Path(song["audio_path"]),
                Path(song["midi_path"]),
                song["split"],
            )
            for song in resolve_catalog_task_manifest_songs(manifest, args.catalog_root)
        ]
    else:
        instrument = "guitar"
        candidates = [
            (
                f"{path.parent.parent.name}__{path.parent.name}",
                path,
                path.parent / "notes.mid",
                "legacy",
            )
            for path in args.songs_root.rglob("guitar.ogg")
            if (path.parent / "notes.mid").exists()
        ]
    candidates.sort(key=lambda candidate: candidate[0])
    if args.max_songs:
        candidates = candidates[: args.max_songs]
    log.info(f"Found {len(candidates)} candidate songs")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    payloads = build_payloads(
        candidates,
        cache_dir=args.cache_dir,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        min_note_length=args.min_note_length,
        instrument=instrument,
    )

    n_ok = n_err = n_cached = n_skip = 0
    t0 = time.time()
    if args.workers <= 1:
        for p in payloads:
            r = process_song(p)
            _tally(r, log)
            n_ok += r["status"] == "ok"
            n_err += r["status"].startswith("error")
            n_cached += r["status"] == "cached"
            n_skip += r["status"] in ("missing", "empty", "no_match")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_song, p) for p in payloads]
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                _tally(r, log)
                n_ok += r["status"] == "ok"
                n_err += r["status"].startswith("error")
                n_cached += r["status"] == "cached"
                n_skip += r["status"] in ("missing", "empty", "no_match")
                if i % 25 == 0:
                    log.info(
                        f"  progress: {i}/{len(futures)}  ok={n_ok} err={n_err} cached={n_cached} skip={n_skip}"
                    )
    log.info(
        f"Done in {time.time() - t0:.1f}s — ok={n_ok} err={n_err} cached={n_cached} skip={n_skip}"
    )


def _tally(r, log):
    if r["status"] == "ok":
        log.info(f"  ✓ {r['song']}: n={r['n']} bp_t={r.get('bp_t', 0):.1f}s")
    elif r["status"].startswith("error"):
        log.warning(f"  ✗ {r['song']}: {r['status']}")


if __name__ == "__main__":
    main()
