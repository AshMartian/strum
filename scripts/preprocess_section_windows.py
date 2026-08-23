"""
Preprocess section-classifier windows.

Reads configs/guitar_section_labels.json and extracts a 2-s log-mel patch per
window from each song's audio, saving to a memmap cache.

Output:
    {cache}/{split}_section_mel.npy   (N, n_mels, T)  fp32
    {cache}/{split}_section_label.npy (N,)            int8
    {cache}/{split}_section_meta.json (song_id, t_start_s for each)

Run:
    python scripts/preprocess_section_windows.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_REPOSITORY_ROOT = _SCRIPTS.parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from src.catalog_task_manifest import resolve_catalog_task_manifest_songs  # noqa: E402
from src.section_frontend import (  # noqa: E402
    N_MELS,
    SAMPLE_RATE,
    WINDOW_FRAMES,
    WINDOW_SAMPLES,
    compute_router_log_mel,
    load_router_audio,
    router_patch_from_log_mel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("section_pre")

LABELS = ["silence", "constant_strum", "chord_stab", "lead_line", "single_notes", "mixed"]
LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}


def process_split(records: list[dict], split: str, cache_dir: Path) -> None:
    # Group by audio path so we load each file once
    by_audio: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["split"] != split:
            continue
        by_audio[r["audio_path"]].append(r)

    if not by_audio:
        log.warning("no records for split=%s", split)
        return

    n_total = sum(len(v) for v in by_audio.values())
    log.info("split=%s songs=%d windows=%d", split, len(by_audio), n_total)

    mel_path = cache_dir / f"{split}_section_mel.npy"
    lab_path = cache_dir / f"{split}_section_label.npy"
    meta_path = cache_dir / f"{split}_section_meta.json"

    mel_mm = np.lib.format.open_memmap(
        mel_path,
        mode="w+",
        # Keep full float32 router features.  fp16 cache compression would
        # turn an otherwise exact declared frontend into a different numeric
        # input at training time.
        dtype=np.float32,
        shape=(n_total, N_MELS, WINDOW_FRAMES),
    )
    lab_mm = np.zeros(n_total, dtype=np.int8)
    meta: list[dict] = []

    cur = 0
    for ai, (audio_path, recs) in enumerate(by_audio.items()):
        if ai % 100 == 0:
            log.info("[%d/%d] %s", ai, len(by_audio), Path(audio_path).name)
        try:
            # This is intentionally the router's own decoder/resampler, not
            # the generic torchaudio data path.  The full-song mel is also
            # computed once before windows are sliced, matching inference.
            audio = load_router_audio(Path(audio_path))
        except Exception as exc:
            log.warning("load failed %s: %s", audio_path, exc)
            continue
        if len(audio) == 0:
            continue
        log_mel = compute_router_log_mel(audio)

        for r in recs:
            t_start = float(r["t_start_s"])
            s = round(t_start * SAMPLE_RATE)
            e = s + WINDOW_SAMPLES
            if e > len(audio):
                continue
            mel_mm[cur] = router_patch_from_log_mel(log_mel, s)
            lab_mm[cur] = LABEL_TO_IDX[r["label"]]
            meta.append(
                {
                    # Catalog-derived labels deliberately carry source IDs,
                    # never the legacy folder-manifest song_id field.
                    "source_id": r["source_id"],
                    "t_start_s": t_start,
                    "label": r["label"],
                }
            )
            cur += 1

    # Trim arrays to actual size
    mel_mm.flush()
    del mel_mm
    if cur < n_total:
        # Re-open to truncate
        full = np.load(mel_path, mmap_mode="r")
        trimmed = np.array(full[:cur])
        np.save(mel_path, trimmed)
        log.info("trimmed mel array %d -> %d", n_total, cur)
        del full, trimmed
    np.save(lab_path, lab_mm[:cur])
    meta_path.write_text(json.dumps(meta))

    counter = Counter(LABELS[label_index] for label_index in lab_mm[:cur])
    log.info("split=%s saved=%d labels=%s", split, cur, dict(counter))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="configs/guitar_section_labels.json")
    ap.add_argument("--cache-dir", default="/mnt/ml-data/guitar_section_cache")
    ap.add_argument(
        "--catalog-manifest", type=Path, help="catalog task manifest used to create labels"
    )
    ap.add_argument(
        "--catalog-root", type=Path, help="catalog root for runtime-only asset resolution"
    )
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.labels).read_text())
    using_catalog = args.catalog_manifest is not None or args.catalog_root is not None
    if using_catalog and (args.catalog_manifest is None or args.catalog_root is None):
        ap.error("--catalog-manifest and --catalog-root must be used together")
    if using_catalog:
        manifest = json.loads(args.catalog_manifest.read_text(encoding="utf-8"))
        task = manifest.get("task")
        if not isinstance(task, dict) or task.get("kind") not in {"section_guitar", "section_bass"}:
            ap.error("catalog manifest must use section_guitar or section_bass")
        resolved = {
            song["source_id"]: song["audio_path"]
            for song in resolve_catalog_task_manifest_songs(manifest, args.catalog_root)
        }
        if data.get("format") != "strum-section-labels/v1":
            ap.error("catalog section preprocessing requires strum-section-labels/v1")
        data["records"] = [
            {**record, "audio_path": resolved[record["source_id"]]}
            for record in data.get("records", [])
            if record.get("source_id") in resolved
        ]
    records = data["records"]
    log.info("loaded %d records", len(records))

    for split in ("val", "test", "train"):
        process_split(records, split, cache_dir)

    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
