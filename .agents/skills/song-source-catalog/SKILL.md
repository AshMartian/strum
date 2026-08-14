---
name: song-source-catalog
description: Validate and consume an OCTAVE song-source catalog for STRUM training. Use when building STRUM datasets, task views, fine-tuning inputs, or evaluation manifests from `octave-song-source-catalog/v1`, including any work involving instrument coverage, audio roles, training rights, or source provenance boundaries.
---

# STRUM Song Source Catalog

Consume OCTAVE's completed `octave-song-source-catalog/v1` as the sole source boundary for training tasks.

## Boundaries

- Use `src.song_source_catalog.load_catalog()` to validate a catalog and every managed asset before reading it.
- Use `select_training_sources()` to make sorted, instrument-specific task inputs. Select only `training_use: allowed` records.
- Keep `source_id` and asset hashes in training/evaluation metadata. Do not retain original package paths, URLs, raw importer errors, provenance/license text, or private catalog-side resolvers.
- Do not add `.sng`, RB3CON, ZIP, or song-folder parsing to STRUM. OCTAVE owns those adapters and catalog materialization.
- Treat lower-difficulty generation as a learned task view, not a deterministic conversion algorithm.

## Build a task view

1. Load a catalog directory; do not parse `catalog.json` or `records.jsonl` directly in a trainer.
2. Choose an instrument from the catalog contract and, when needed, a keyed audio role such as `guitar`, `vocals`, or `mix`.
3. Call `select_training_sources(catalog, instrument, required_difficulties=..., audio_role=...)`.
4. Convert only the returned MIDI/audio paths into task-specific windows, tokens, targets, splits, and evaluation manifests.
5. Record catalog ID, source IDs, selected asset hashes, task-builder version, split assignment, and model/checkpoint revision. Never duplicate rights text or absolute source paths.

```python
from src.song_source_catalog import load_catalog, select_training_sources

catalog = load_catalog(catalog_root)
sources = select_training_sources(
    catalog, "guitar", required_difficulties=("expert",), audio_role="guitar"
)
```

## Validate before training

```bash
VENV_PY=/run/media/ash/portable-ai/strum/venv/bin/python
"$VENV_PY" -m pytest tests/test_song_source_catalog.py -q
"$VENV_PY" -m src.song_source_catalog /path/to/new-catalog --instrument guitar
```

Fail closed on missing assets, hash mismatches, malformed records, unsupported fields, unapproved rights, or unsafe path/URI text. Ask OCTAVE to rebuild invalid catalogs; do not repair them in STRUM.
