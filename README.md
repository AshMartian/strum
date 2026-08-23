<p align="center">
  <img src="assets/logo.png" alt="STRUM Logo" width="200"/>
</p>

<h1 align="center">STRUM</h1>
<h3 align="center"><b>S</b>pectral <b>T</b>ranscription & <b>R</b>hythm <b>U</b>nderstanding <b>M</b>odel</h3>

<p align="center">
  AI-powered audio-to-chart pipeline for Clone Hero & YARG
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="PyTorch 2.x"/>
  <img src="https://img.shields.io/badge/CUDA-12.8-76b900.svg" alt="CUDA 12.8"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/>
</p>

---

STRUM is an audio-to-chart research runtime for Clone Hero / YARG. Its legacy
batch scripts can assemble multi-instrument chart packages, while its versioned
worker exposes only explicitly declared, bundle-validated inference profiles.
Today those executable worker profiles are Expert Guitar (`guitar.hybrid-v2-rule/v1`),
Expert Drums through the direct V14 interpreter (`drums.v14-expert/v1`), and
learned five-lane difficulty transforms (`difficulty.transform/v1`). Bass,
vocals, keys, and Pro-instrument behavior in the legacy scripts are not yet
deployable worker capabilities.

The system includes a two-stage neural drum transcription pipeline, hybrid
Guitar transcription, experimental legacy Bass/Vocals/Keys paths, and
catalog-backed training adapters. Lower difficulties are a STRUM concern: a
worker run is Expert-only unless an explicit, validated STRUM difficulty profile
is selected. OCTAVE must never supply its own deterministic Expert-to-lower-
difficulty mapping.

## Architecture

```
                              ┌─────────────┐
                              │  Audio File  │
                              │  (WAV/MP3)   │
                              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │   Demucs v4  │
                              │  Separation  │
                              └──────┬───────┘
                                     │
              ┌──────────┬───────────┼───────────┬──────────┐
              ▼          ▼           ▼           ▼          ▼
         ┌────────┐ ┌────────┐ ┌─────────┐ ┌────────┐ ┌────────┐
         │ Drums  │ │ Guitar │ │  Bass   │ │ Vocals │ │  Keys  │
         │  Stem  │ │  Stem  │ │  Stem   │ │  Stem  │ │ Other  │
         └───┬────┘ └───┬────┘ └────┬────┘ └───┬────┘ └───┬────┘
             │          │           │           │          │
             ▼          ▼           ▼           ▼          ▼
        ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
        │Two-Stage│ │ Neural  │ │ Neural  │ │ Whisper │ │Spectral │
        │  CRNN   │ │ Onset + │ │ Onset + │ │ + pYIN  │ │Keyboard │
        │Ensemble │ │Rule Fret│ │Rule Fret│ │ + Align │ │Detector │
        └───┬─────┘ └───┬─────┘ └───┬─────┘ └───┬─────┘ └───┬─────┘
             │          │           │           │          │
             └──────────┴───────────┼───────────┴──────────┘
                                    ▼
                          ┌──────────────────┐
                          │   Chart Export    │
                          │  .mid + song.ini │
                          │  + album art     │
                          │  (4 difficulties)│
                          └──────────────────┘
```

## Instrument Pipelines

### Drums — Two-Stage Neural Ensemble

The drums pipeline is the flagship component, using a two-stage detection-then-classification approach:

1. **Onset Detection** — V14 `TwoStageDrumsCRNN` processes mel spectrograms (128 bins, 22050 Hz) to detect drum hit positions with **93.9% F1 score**
2. **Ensemble Classification** — 6 independently trained `OnsetClassifier` models (V2, V4, V6, V12c, V15, V16) vote on each detected onset to classify across 8 lanes (Kick, Snare, Hi-Hat, Crash, Ride, High Tom, Mid Tom, Floor Tom) achieving **85.2% F1 score**
3. **Spectral Disambiguation** — Spectral centroid analysis resolves tom/cymbal confusion in ambiguous frequency ranges
4. **Post-Processing** — Bidirectional iterative streak smoothing, kick-suppresses-floor-tom logic, rhythmic quantization, and lane conflict resolution

Pro drums are fully supported with separate tom and cymbal markers per the Clone Hero MIDI specification.

### Guitar & Bass — Neural Onset + Polyphonic Pitch + Fret Mapping

Guitar and bass share the same hybrid architecture (`src/inference/guitar_hybrid_v2.py`):

1. **Onset Detection** — `OnsetCRNN` (V2) detects note attacks on the Demucs-separated stem so vocals/drums don't trigger false positives.
2. **Polyphonic Pitch** — [Spotify Basic Pitch](https://github.com/spotify/basic-pitch) transcribes simultaneous notes (chords + single notes), with bass-specific MIDI range overrides (24–67) when running on the bass stem.
3. **Pitch → Fret Mapping** — Rule-based register allocation by default. The historic
   learned mapper is a research-only compatibility path enabled with
   `STRUM_GUITAR_FRET_MAPPER=learned`; it is not selectable as a STRUM model-bundle
   profile, and an explicit request fails rather than silently using rules.
4. **Section-Aware Density** — An optional `SectionRouter` modulates onset peak thresholds per section (verse/chorus/solo) to prevent over- or under-charting.

### Vocals — Whisper + pYIN Pitch Tracking

1. **Lyric Transcription** — OpenAI Whisper extracts word-level timestamps from the vocal stem
2. **Pitch Detection** — `librosa.pyin` tracks vocal pitch contours at high time resolution
3. **Dynamic Alignment** — Whisper word boundaries are aligned with pitch onsets for accurate note placement
4. **Lyrics Fetching** — Optional synced lyrics from LRCLIB and Lyrics.ovh APIs
5. **Harmony Detection** — Configurable threshold (default 30%) for harmony/backing vocal phrases

### Keys — Spectral Keyboard Detection

1. **Keyboard Detection** — Spectral flatness and harmonic ratio analysis identifies keyboard-active regions in the "other" stem
2. **Note Extraction** — `librosa.onset_detect` + `librosa.piptrack` extract individual key hits
3. **Dual Output** — Both 5-lane simplified and Pro Keys (full piano range) tracks

## Tempo & Grid Alignment

STRUM uses a grid-alignment BPM refinement algorithm that searches ±5 BPM around an initial `librosa` estimate at 0.1 BPM resolution, then snaps the beat-zero phase to the first detected onset. Phase coherence is measured with circular statistics on beat positions vs. onset times. After all transcribers run, every event from every instrument is shifted by the same `phase_offset_ms` and snapped to the 32nd-note grid — with per-lane roll detection to preserve fast double-strokes and tom rolls.

This reduces post-snap grid error to <5 ms on the verified test set across drums, guitar, bass, vocals, and keys.

## Performance

### Component-level (held-out test set)

| Component | Metric | Score |
|-----------|--------|-------|
| Drums — Onset Detection (V14) | Frame F1 | 93.9% |
| Drums — Lane Classification (6-model ensemble) | Per-onset F1 | 85.2% |
| Drums — Best Single Classifier (V12c) | Per-onset F1 | 83.8% |

Evaluated on a held-out test set from 3,299 human-authored Clone Hero/YARG pro drum charts.

### End-to-end vs human-authored game charts (in-envelope benchmark, n=29)

Aggregate per-instrument onset F1 against ground-truth Clone Hero/YARG charts. Songs were sampled from a held-out pool of 3,299 candidates and pre-screened by a single audio-feature operating envelope: median Demucs `htdemucs_6s` drum-stem RMS (1-second windows, 22050 Hz mono) ≥ 0.018. Eval is Expert difficulty, ±100 ms tolerance with a per-song global offset search (±200 ms / 10 ms steps) to neutralize chart-sync conventions.

| Instrument | F1 | Precision | Recall |
|------------|------|-----------|--------|
| Drums      | 83.8% | 82.4% | 85.4% |
| Guitar     | 65.1% | 74.5% | 57.8% |
| Bass       | 69.4% | 65.8% | 73.4% |
| Vocals     | 53.9% | 63.2% | 47.0% |

Reproduce with:

```bash
python scripts/eval_benchmark.py \
  --gt-dir /path/to/charts-gt \
  --pred-dir /path/to/strum-predictions \
  --tolerance-ms 100 \
  --global-offset-search \
  --out benchmark_results.json
```

## Quick Start

### Prerequisites

- Python 3.11+
- PyTorch 2.x with CUDA
- ffmpeg
- ~6 GB disk for model checkpoints

### Installation

```bash
git clone https://github.com/oprialopez/strum.git
cd strum
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Model checkpoints (~6 GB) are not committed; download from the releases page or train locally (see below).

### Generate Charts for a Song

Drop one or more `.wav` / `.mp3` / `.flac` files in a directory and run:

```bash
# Legacy full chart package (not a worker deployment profile)
python scripts/batch_pipeline.py \
  --songs-dir /path/to/songs/ \
  --output-dir /path/to/output/

# Drums only (faster, no Demucs vocals/keys/bass passes)
python scripts/batch_infer_hybrid.py \
  --songs-dir /path/to/songs/ \
  --output-dir /path/to/output/
```

Each output folder contains `notes.mid`, `song.ini`, the source audio, and album art ready to drop into Clone Hero / YARG.

#### Backend selection (env vars)

| Variable | Values | Default | Effect |
|----------|--------|---------|--------|
| `STRUM_GUITAR_BACKEND` | `hybrid`, `neural`, `rule`, `basicpitch` | `hybrid` | Guitar transcription pipeline |
| `STRUM_BASS_BACKEND`   | `hybrid`, `neural`, `rule`, `basicpitch` | `hybrid` | Bass transcription pipeline |
| `STRUM_GUITAR_FRET_MAPPER` | `rule`, `learned` | `rule` | Legacy research mapper only; `learned` fails closed if unavailable |
| `STRUM_V12C_VARIANT`   | `default`, `community` | `default` | Swap drum classifier v12c checkpoint |

### Legacy script training

These script entry points predate the worker contract. They are useful for
research and for the catalog adapters described below, but a script-produced
checkpoint is not automatically an OCTAVE-deployable profile. Deployable
checkpoints require a validated STRUM model bundle and profile manifest.

| Model | Preprocess | Train | Config |
|-------|------------|-------|--------|
| Drum onset detector (V14 CRNN) | `preprocess_onset_windows.py` | `python scripts/train_onset_classifier.py` (also trains the onset head) | `configs/drums_v14.yaml` |
| Drum classifier ensemble (V2/V6/V12c/V15/V16) | `preprocess_onset_windows.py` | `python scripts/train_onset_classifier.py --config configs/onset_classifier_v15.yaml` | `configs/onset_classifier*.yaml` |
| Tom-vs-cymbal refinement | (uses Demucs drum stem at train time) | `python scripts/train_tom_refinement.py` | inline |
| Guitar onset CRNN (V1/V2) | `build_guitar_manifest.py` → `preprocess_guitar_windows.py` | `python scripts/train_guitar_v1.py --config configs/guitar_v2.yaml` | `configs/guitar_v1.yaml`, `configs/guitar_v2.yaml` |
| Pitch→fret mapper | `build_mapper_dataset.py` | `python scripts/train_fret_mapper.py` | inline |
| Section classifier (verse/chorus/etc.) | `build_section_labels.py` → `preprocess_section_windows.py` | `python scripts/train_section_classifier.py` | inline |

A typical training session for the drum onset detector looks like:

```bash
python scripts/preprocess_onset_windows.py \
  --manifest /mnt/ml-data/manifest.json \
  --output-dir /mnt/ml-data/onset_windows/

python scripts/train_onset_classifier.py \
  --config configs/onset_classifier_v15.yaml
```

W&B logging is enabled by default; set `WANDB_MODE=offline` to disable.

## Project Structure

```
strum/
├── configs/                          # YAML configs (one per trainable model)
│   ├── drums_v14.yaml                # Two-stage drum onset CRNN
│   ├── onset_classifier_v{6,12_clean,15,16}.yaml  # Drum classifier ensemble
│   ├── guitar_v1.yaml, guitar_v2.yaml             # Guitar onset CRNN
│   ├── inference.yaml, preprocessing.yaml
├── checkpoints/                      # Trained weights (gitignored)
├── scripts/
│   ├── batch_pipeline.py             # ★ Full multi-instrument pipeline (entry point)
│   ├── batch_infer_hybrid.py         # Legacy Drums batch pipeline
│   ├── chart_postprocess.py          # Snap-to-grid + rescue passes + quantization
│   ├── chart_enhancer.py             # Difficulty reduction + lane balancing
│   ├── vocals_charter.py             # Whisper + pYIN vocal transcription
│   ├── keys_charter.py               # Keyboard detection + Pro Keys export
│   ├── guitar_basicpitch.py          # Basic-Pitch guitar backend
│   ├── bass_basicpitch.py            # Basic-Pitch bass backend
│   ├── train_onset_classifier.py     # Drum classifier training
│   ├── train_tom_refinement.py       # Tom-vs-cymbal refinement training
│   ├── train_guitar_v1.py            # Guitar onset CRNN training
│   ├── train_fret_mapper.py          # Learned pitch→fret mapper training
│   ├── train_section_classifier.py   # Section (verse/chorus/etc.) training
│   ├── preprocess_onset_windows.py   # Drum window preprocessing
│   ├── preprocess_guitar_windows.py  # Guitar window preprocessing
│   ├── preprocess_section_windows.py # Section window preprocessing
│   ├── build_guitar_manifest.py      # Guitar dataset manifest builder
│   ├── build_mapper_dataset.py       # Pitch→fret dataset builder
│   └── build_section_labels.py       # Section label builder
├── src/
│   ├── models/
│   │   ├── drums_v13.py              # TwoStageDrumsCRNN architecture (V14 ckpt)
│   │   ├── onset_classifier.py       # 8-lane drum classifier
│   │   ├── onset_classifier_dataset.py, onset_classifier_cached_dataset.py
│   │   ├── drums_v14_dataset.py      # Drum onset dataset w/ bg-mel subtraction
│   │   ├── tom_refinement.py         # Tom-vs-cymbal head
│   │   ├── guitar_v1.py              # Guitar onset CRNN
│   │   ├── section_classifier.py     # Section labeler
│   │   ├── bg_mel.py                 # Background-mel subtraction
│   │   └── common.py
│   ├── inference/
│   │   ├── guitar_hybrid_v2.py       # Hybrid Guitar/Bass research backend
│   │   ├── guitar_neural.py          # Neural-only guitar/bass backend
│   │   ├── guitar_bass.py            # GuitarChart/Note/Chord dataclasses + rule backend
│   │   ├── section_router.py         # Section-aware onset gating
│   │   └── c3_rules.py               # C3 chart rules (5-fret reduction etc.)
│   ├── preprocessing/
│   │   ├── parsers/                  # .mid and .chart parsers
│   │   ├── alignment.py              # Audio-chart alignment
│   │   └── separation.py             # Demucs wrapper
│   ├── export/
│   │   ├── midi.py                   # Pro drums + guitar/bass/keys MIDI export
│   │   └── chart.py                  # .chart format export
│   └── lyrics/                       # LRCLIB + Lyrics.ovh fetcher
├── docs/
│   ├── ARCHITECTURE.md               # Technical specification
│   └── ROADMAP.md                    # Development milestones
└── pyproject.toml
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| ML Framework | PyTorch 2.x |
| Audio Separation | Demucs v4 (HTDemucs) |
| Pitch Detection | librosa pYIN |
| Speech-to-Text | OpenAI Whisper |
| MIDI I/O | mido |
| Experiment Tracking | Weights & Biases |
| Config Management | Hydra |
| Audio Processing | librosa, soundfile |
| CLI | Click + Rich |

## Chart Output Format

STRUM generates standard Clone Hero / YARG compatible chart packages:

```
Song Name/
├── notes.mid          # MIDI chart (480 ticks/beat, 4 difficulty levels)
├── song.ini           # Metadata (artist, title, charter, BPM)
├── song.ogg           # Audio file
└── album.png          # Album art (fetched automatically)
```

Each MIDI contains up to 5 instrument tracks:
- **PART DRUMS** — 5-lane pro drums with cymbal markers (MIDI notes 96-100, tom markers 110-112)
- **PART GUITAR** — 5-fret guitar (MIDI notes 96-100)
- **PART BASS** — 5-fret bass (MIDI notes 96-100)
- **PART VOCALS** — Pitched vocal phrases with lyric events
- **PART KEYS** — 5-lane keys + optional Pro Keys

Four difficulty levels per instrument: Expert, Hard, Medium, Easy (progressive note reduction).

## Custom model bundles

The repository `checkpoints/` layout remains the default. To evaluate or ship
a fine-tuned model without replacing it, set `STRUM_MODEL_BUNDLE` to a
directory containing `strum-model-bundle.json` (or directly to that file).
Each path is relative to the manifest, so a bundle can be copied as one
directory. Components not declared by a bundle keep using the repository
defaults, which makes small targeted experiments safe.

```json
{
  "schema_version": 1,
  "model_id": "drums-v14-finetune-2026-08",
  "compatibility": {
    "manifest_schema": 1,
    "strum_version": ">=0.1.0",
    "strum_revision": "<pinned-strum-git-revision>"
  },
  "components": {
    "drums.v14_onset": {
      "checkpoint": "weights/best.pt",
      "sha256": "<64-character lowercase sha256>"
    },
    "drums.ensemble.v17": {
      "checkpoint": "weights/v17-best.pt",
      "config": "configs/onset_classifier_v17.yaml"
    },
    "guitar.onset": {"checkpoint": "weights/guitar-onset.pt"}
  }
}
```

Validate a candidate without loading model weights, or list valid bundles in
a user model directory:

```bash
python -m src.model_bundle validate /path/to/bundle --check-files --verify-hashes
python -m src.model_bundle list /path/to/models
```

For OCTAVE or another host, use the worker discovery boundary instead of
walking that folder or deriving profile support from filenames:

```bash
strum-worker checkpoint discover --model-root /private/models --json
strum-worker checkpoint inspect --model-root /private/models/release-a --json
```

`checkpoint discover` recursively scans one bounded user-selected folder
(eight levels, at most 256 manifests), validates component hashes without
loading tensors, and returns only opaque manifest-derived `artifact_id` values,
model/profile identities, component hashes, and capability status. It never
returns the selected folder, manifest location, or checkpoint location. Hosts
keep the `artifact_id` → local-folder mapping in their main process. A profile
is `execution.available` only when its declared component hashes *and* its
Guitar/Bass/Keys/Drums/transform-specific configuration contract pass; a
hash-valid experiment or a profile without a STRUM chart handler remains
`not_deployable`. Selecting a candidate still requires `inference profile
validate` and chart preflight for the explicit instruments and policy.

The current model-bundle validator recognizes `drums.v14_onset`,
`drums.ensemble.v2` through `drums.ensemble.v17`, and `guitar.onset`.
`compatibility.strum_revision` is optional but recommended for portable
bundles: it records the STRUM Git revision the model was trained against. Set
`STRUM_SOURCE_REVISION` to the source revision pinned by a caller (such as an
editor integration) to enforce it. Otherwise validation reports the declared
revision as unverified, rather than treating source trees without Git metadata
as incompatible.

### Chart-pair fine-tuning prototype

`scripts/train_chart_transform.py` is a small CPU/CUDA baseline for learned
chart-to-chart stages, including Expert → lower-difficulty experiments. It
uses local paired chart events and a song-level split. It can also condition
each source event on a local song through a small RMS/transient feature vector;
it never packages song audio or paths into the dataset export or model bundle.
Every dataset needs a `dataset-manifest.json` containing a
non-empty `provenance` and `license`, plus JSONL pairs in the
`strum-chart-pairs/v1` format. Copy
`configs/chart_transform_finetune.yaml`, set the local paths, then run:

```bash
python scripts/train_chart_transform.py --config /path/to/experiment.yaml
```

Set `device: auto` to train on `cuda:0` when CUDA is available (otherwise CPU),
or set `device: cpu`, `device: cuda`, or `device: cuda:<index>` explicitly. The resolved device
and CUDA adapter name are recorded in `training-metadata.json`; an explicit
CUDA request fails clearly when that device is unavailable. The small baseline
is intended to validate data and bundle plumbing, so meaningful GPU utilization
requires a larger windowed dataset and sequence model.

The dataset manifest and each JSONL record are intentionally small and
portable:

```jsonc
// dataset-manifest.json
{
  "schema_version": 1,
  "format": "strum-chart-pairs/v1",
  "dataset_id": "my-authorized-chart-pairs",
  "records": "pairs.jsonl",
  "provenance": "where and how the paired charts were obtained",
  "license": "license or permission covering model-training use"
}
// pairs.jsonl (one JSON object per line)
{
  "song_id": "stable-song-id",
  "source_difficulty": "Expert",
  "target_difficulty": "Hard",
  "source_events": [{"time_ms": 1000, "lanes": [0, 2]}],
  "target_events": [{"time_ms": 1000, "lanes": [0]}]
}
```

`lanes` are zero-based 5-fret/lane indices by default. A target event is
matched to its nearest Expert event within `alignment_tolerance_ms`; unmatched
Expert events learn the all-off target. This deliberately modest baseline does
not yet model target-only inserted notes or ergonomic sequence decisions.

For standalone scripts, keep the private audio mapping separate from the
portable chart-pair dataset. Set `audio_feature_mode: rms_onset_v1` and
`audio_manifest` in the training config (or pass both
`--audio-feature-mode rms_onset_v1 --audio-manifest ...`). Paths are relative
to the audio manifest, so it can live next to a private local song library
without copying audio to the dataset or bundle:

```json
{
  "schema_version": 1,
  "format": "strum-local-audio-assets/v1",
  "assets": [
    {"song_id": "stable-song-id", "audio": "song.ogg"}
  ]
}
```

For the worker/OCTAVE path, select `rms_onset_v1` during **Prepare** instead.
STRUM records only selected catalog audio roles and hashes in the immutable
task view, excludes chart/audio duration mismatches, and creates a short-lived
worker-private manifest during Train. Paths and temporary audio copies never
enter task views, experiment metadata, model bundles, or worker results.

`ffmpeg` decodes local audio to bounded mono PCM for this prototype. The
checkpoint records the feature schema and requires the same `--song` during
inference:

```bash
python scripts/infer_chart_transform.py \
  --checkpoint /path/to/bundle/weights/chart_transform.pt \
  --source-events /path/to/expert-events.json \
  --song /path/to/local/song.ogg \
  --output /path/to/hard-events.json
```

This is an audio-conditioned event baseline, not a learned audio encoder. It
is useful for validating the data contract and song alignment; the next model
should use guitar/stem-aware temporal windows and report a held-out,
song-disjoint comparison against the chart-only baseline.

To prepare authorized, extracted Clone Hero/YARG charts locally, use the
five-lane bridge on directories of `notes.mid` files (or pass a text list with
`--list-file`). Build one dataset per instrument:

```bash
python scripts/prepare_instrument_chart_pairs.py \
  --input /path/to/extracted-song-folders \
  --output-dir /path/to/local/guitar-expert-hard \
  --instrument guitar \
  --target-difficulty Hard \
  --dataset-id my-guitar-pairs-v1 \
  --provenance "authorized local chart export" \
  --license "permission recorded by the dataset owner"
```

It supports `PART GUITAR`, `PART BASS`, `PART KEYS`, and `PART DRUMS` through
the same standard 5-lane MIDI ranges (Expert 96–100; Hard 84–88; Medium 72–76;
Easy 60–64), and writes `pairs.jsonl` plus `dataset-manifest.json`. Each
dataset and resulting component is instrument-labelled, so a Bass model cannot
be mistaken for Guitar. Open notes and modifier notes are excluded from this
first five-lane baseline. Song IDs combine a sanitized parent-folder label and
a content hash, so output records do not reveal absolute input paths. Existing
dataset files are preserved unless `--overwrite` is explicitly supplied.

For compatibility with the earlier Guitar-only prototype, an old
`strum-chart-pairs/v1` manifest or record without `instrument` is treated as
Guitar and the resulting component is labelled accordingly. New exports always
write the explicit instrument label.

`PART VOCALS` and Pro Guitar/Pro Keys deliberately do not pass through this
five-lane bridge: vocals are pitch/phrase/lyric sequences and Pro instruments
use fret/string semantics. They need their own event schemas and losses, not a
lossy conversion to five lanes. The original
`prepare_guitar_chart_pairs.py` remains a compatible Guitar-default alias.

The output is a registry-valid bundle plus reproducibility config, split IDs,
provenance/license, alignment counts, and validation metrics. Direct script
configuration may set `init_checkpoint` to a compatible prior
`EventTransformMLP` checkpoint; it is always read with PyTorch's tensor-only
loader. The worker contract is stricter: fine-tuning can select only a
hash-verified STRUM model bundle, never an arbitrary checkpoint path. STRUM
releases weights and a benchmark manifest, not its original community-chart
training corpus; supply only chart pairs you are authorized to use.

## OCTAVE song-source catalogs

STRUM treats OCTAVE's `octave-song-source-catalog/v1` as its common local
training-source boundary. OCTAVE imports source packages, makes the rights
decision, and materializes managed assets. STRUM then validates asset hashes
and selects only `training_use: allowed` records for a task view; it does not
parse `.sng`, `.rb3con`, ZIP, or source-folder packages, or persist original
source paths.

### Worker contract

OCTAVE calls the installed `strum-worker` interface (or `python -m src.worker`
for a managed developer checkout), never individual `scripts/*.py` files. The
worker reports the distinction between a pipeline whose task view can be built
and one whose trainer can be run through the worker:

```bash
strum-worker probe --json
strum-worker pipeline list --json
strum-worker catalog inspect --catalog-root /path/to/catalog --pipeline guitar.onset-fret/v1 --json
strum-worker catalog inspect --catalog-root /path/to/catalog --pipeline chart_transform.five_lane/v1 --options '{"instrument":"guitar","target_difficulty":"Hard"}' --json
strum-worker dataset prepare --request /path/to/owned-prepare-request.json --json
```

`catalog inspect` is the UI planning preflight: a selected pipeline returns an
`eligible_count`, stable safe exclusion-code counts, its effective audio
policy, and a bounded `estimated_storage_bytes`.  The estimate is the sum of
distinct immediate catalog input assets; it deliberately excludes generated
task views, preprocessing caches, and checkpoints.  Supply the optional
selection subset with `--options` whenever a preparation choice changes
eligibility (in particular, chart-transform instrument and target difficulty).
The result never includes catalog/source paths, record IDs, rights text, or
provenance.

For OCTAVE-supervised background work, use line-delimited lifecycle events
instead of parsing human output:

```bash
strum-worker dataset prepare --request /path/to/owned-prepare-request.json --json-events
strum-worker train start --request /path/to/owned-train-request.json --json-events
```

Each stream has an opaque request-derived job ID, monotonic sequence numbers,
stage/progress, and a `succeeded` or `failed` terminal state. OCTAVE owns the
child process group and cancellation; STRUM does not run a separate mutable
job daemon that could retain private paths after the supervising process exits.

The request is a main-process-only file; paths are not echoed in the response.
For example, a Guitar task-view request is:

```json
{
  "catalog_root": "/private/catalog",
  "pipeline_id": "guitar.onset-fret/v1",
  "output": "/private/task-views/guitar-v1.json",
  "options": {"required_difficulty": "expert"}
}
```

`guitar.onset-fret/v1`, `bass.onset-fret/v1`, `keys.onset-fret/v1`,
`drums.onset-classifier/v1`, `vocals.note-activity/v1`,
`vocals.phrase-boundaries/v1`,
`chart_transform.five_lane/v1`, and the separate
`strum.fret-mapper/guitar/v1` / `strum.fret-mapper/bass/v1` derived-label
pipelines are worker-trainable. Fret-mapper training builds Basic-Pitch
features only from its revalidated approved task view, preserves the catalog's
song-level train/validation split, requires the STRUM `pitch` extra, and records
the exact `basic-pitch` distribution version that emitted the feature data. It
produces an experiment component, not an auto-chart profile. The immutable
experiment release requirements block promotion until a profile composes a
verified onset source, tensor-only loader, pinned Basic-Pitch/Viterbi policy,
and an end-to-end held-out chart evaluation. Their renderer-visible
schemas contain only bounded model/training knobs. The private top-level
`catalog_root` request field is worker-local configuration for Guitar, Bass,
Keys, Drums, Vocals, and fret-mapper task-view revalidation, never a pipeline
option or renderer control.

Guitar invokes the established window-preprocessing and two-stage onset/fret
trainers, then packages verified `guitar.onset` and `guitar.fret` bundle
components with task-view lineage. Like the Drums path, its output is an
experiment artifact (`deployment_status: requires_profile_packaging`): it
cannot replace a production auto-chart profile until a compatible profile is
explicitly packaged and validated. Drums derives its eight-lane labels,
builds the maintained onset-window cache, and invokes the onset-classifier
trainer. Its output remains an experiment artifact
(`deployment_status: requires_profile_packaging`), not a claim that it can
replace the verified `drums.v14-expert/v1` auto-chart profile. The five-lane
transform creates a verified learned lower-difficulty component. OCTAVE must
surface these distinct deployment states rather than selecting a checkpoint
implicitly.

Vocals has separate bounded experiments because `PART VOCALS` is not a
five-lane chart. `vocals.note-activity/v1` revalidates a dedicated
`vocals_activity` task view whose only declared label track is the exact
`PART VOCALS` identity. It trains frame-level pitched-vocal activity plus MIDI
pitch 36--84 from a vocal stem (or approved mix fallback). The companion
`vocals.phrase-boundaries/v1` task derives lead phrase starts/ends only from
the supported MIDI 105/106 marker convention or a sustained 105 span. Its
separate `vocals.phrase_boundaries` component is likewise experiment-only.
Neither path trains word recognition, talkies, harmonies, or a playable chart.
The planned `strum.instrument-chart/vocals/v1` descriptor publishes those
remaining machine-readable stages and rejects chart execution until a composed
Vocal profile has passed held-out chart evaluation and packaging. OCTAVE must
never offer either raw component as an auto-chart model.

Bass invokes the same five-lane CRNN implementation only after STRUM has
revalidated the dedicated `bass_onset_fret` task view and its `PART BASS`
labels. It emits distinct `bass.onset` and `bass.fret` components plus a
`strum-bass-neural-model-config/v1` configuration. Its experiment status is
`requires_bass_profile_evaluation_and_packaging`: that raw artifact cannot be
selected for auto-charting or substituted for Guitar. A separate `bass profile
evaluate` command revalidates held-out `PART BASS` labels and a `bass profile
package` command may then copy it into a hash-verified
`bass.neural-v1-expert/v1` bundle. The deployable profile emits only
`PART BASS` Expert notes; lower difficulties remain an explicit STRUM
difficulty-transform decision.

Pro Guitar, Pro Bass, and Pro Keys are deliberately **not** aliases for those
five-lane experiments. They are catalog-ready only. Their discovered
descriptors carry a `strum-planned-training-contract/v1` with
`training_status: planned`, a machine-readable ordered set of required STRUM
stages, and `execution.status: not_available`. Pro Guitar/Bass task views
select only the exact `PART REAL_GUITAR` / `PART REAL_GUITAR_22` or
`PART REAL_BASS` / `PART REAL_BASS_22` identities. Prepare materializes a
path-free `strum-pro-target-task-manifest/v1`: it decodes Expert string, fret,
duration, technique, and standard-versus-`_22` variant targets without ever
mapping them to five lanes. Pro Keys preparation likewise decodes only
`PART REAL_KEYS_X` pitches, durations, channels, and range shifts; `E`, `M`,
and `H` are not accidental inputs. A malformed or unsupported REAL_* MIDI
source is excluded explicitly. No Pro descriptor has a train schema,
checkpoint profile, inference capability, or chart handler until its audio
preprocessor, event model, held-out evaluator, profile package, and chart
execution stages are implemented and validated.

Keys has the same narrow experiment boundary: `keys.onset-fret/v1` revalidates
the dedicated `keys_onset_fret` task view, whose label schema selects only
`PART KEYS` Expert five-lane labels. It emits distinct `keys.onset` and
`keys.fret` components plus a `strum-keys-neural-model-config/v1`
configuration. Its `deployment_status` is
`requires_keys_profile_evaluation_and_packaging`; raw worker checkpoints cannot
be selected for auto-charting or substituted for Guitar or Bass. A separate
`keys profile evaluate` command revalidates held-out `PART KEYS` labels, and a
`keys profile package` command may only then create an immutable,
hash-verified `keys.neural-v1-expert/v1` profile. That profile writes only
Expert `PART KEYS`; lower difficulties remain an explicit learned STRUM
difficulty-transform decision.

```bash
strum-worker train start --request /path/to/owned-train-request.json --json-events
```

For example, OCTAVE owns these private locations and never displays them from
worker output:

```json
{
  "pipeline_id": "guitar.onset-fret/v1",
  "task_view": "/private/task-views/guitar-v1.json",
  "output": "/private/experiments/guitar-v1",
  "catalog_root": "/private/catalog",
  "options": {
    "model_id": "my-guitar-v1",
    "epochs": 25,
    "batch_size": 128,
    "device": "auto"
  }
}
```

The chart-transform training request names a catalog-generated
`dataset-manifest.json`, an output folder, and bounded configuration such as
`model_id`, `epochs`, `device`, and `hidden_dim`. `checkpoint_mode` is either
`fresh` (the default) or `fine_tune`; `resume` is intentionally rejected until
STRUM has a portable optimizer/scheduler state contract. The renderer exposes
an opaque `parent_artifact_id`, which OCTAVE resolves in its main process. A
`fine_tune` request then supplies the resolved `parent_bundle` as a private
top-level request field, parallel to `catalog_root`; it is not a schema option
and is never renderer-visible. STRUM verifies the bundle manifest, component
hashes and byte lengths, its exact five-lane architecture/preprocessing,
compatible task settings, and expected inference profile before opening its
tensor-only checkpoint. The parent path is never placed in the result,
`training-metadata.json`, or `experiment.json`; those artifacts retain only
parent model/component and manifest/checkpoint hashes.
It does not accept arbitrary checkpoint paths or audio locations. Its
resulting bundle includes a verified component hash/byte length, architecture
and preprocessing IDs, plus a profile that can be checked with
`strum-worker inference profile validate`.
The profile is executable through `strum-worker chart run`: it consumes an
explicit Expert five-lane `notes.mid` and writes only its declared learned
target difficulty. This lets OCTAVE compose an Expert chart stage with an
explicit STRUM difficulty stage, rather than applying a deterministic
downgrade. A transform request is private to the supervising process:

```json
{
  "preflight_request": "/private/transform-preflight.json",
  "source_midi_path": "/private/expert-notes.mid",
  "song_path": null,
  "output_dir": "/private/transform-run",
  "threshold": 0.5
}
```

`song_path` is required only when the selected transform checkpoint is
audio-conditioned. The output contains `events.json`, one target-difficulty
`notes.mid`, and a hash-recorded `run.json`; none contain input locations.

### Typed chart preflight and result contract

`chart preflight` returns `strum-chart-preflight/v1`; a completed chart run
writes `strum-chart-run/v1` to `run.json`. Both contain `instrument_results`
and an explicit `difficulty` object. The command response carries the same
pair under `chart_result`, preserving its pre-existing top-level `difficulty`
target string for older callers. Every requested instrument has typed stage
records with a status, required flag, declared component IDs, and an output
difficulty where one is known. Artifact references are stable output names and
hashes only.

The currently executable Expert Guitar and Expert Drums profiles report an
`expert_chart` stage and a `difficulty_transform` stage marked
`not_requested` with the `expert_only` policy. A learned transform reports
the Expert stage as `provided` (it consumes the caller-provided Expert MIDI)
and its own `difficulty_transform` stage as `ready` at preflight and
`succeeded` after execution. A profile without a matching worker handler is
reported as `not_available`/`unavailable`, rather than being silently routed
through a legacy fallback. This lets OCTAVE distinguish a deliberately
partial result from a complete multi-stage auto-chart graph.

The contract never includes catalog, audio, MIDI, model-root, request, or
output-directory paths.

### Composed profile graphs

`strum-model-bundle.json` profiles may declare a path-free `graph` for a
complete future auto-chart composition. The graph is deliberately separate
from a chart handler: it describes what must be verified, while only STRUM
code that explicitly registers the profile capability may execute it.
`strum-worker checkpoint inspect` exposes these safe profile summaries for
OCTAVE before it offers a profile selector.

Each graph has dependency nodes and terminal chart outputs. A node
declares its stable ID/kind, optional instrument and difficulty, required flag,
bundle component IDs, versioned `companions`, upstream stage IDs, and typed
input/output artifact identities. Inputs beginning `source.` are caller inputs;
every other input must be emitted by a transitive dependency. Terminal outputs
bind a declared instrument, producing stage, artifact identity, and difficulty.
The loader rejects cycles, missing producers, unbound outputs, undeclared
components/companions, and graph outputs that do not exactly cover the profile
instruments.

The bundle may declare non-checkpoint runtime companions such as Demucs,
Basic Pitch, or Whisper in its top-level `companions` object:

```json
{
  "companions": {
    "demucs": { "kind": "runtime", "version": ">=4.0" }
  },
  "profiles": {
    "guitar-composed": {
      "required_companions": ["demucs"],
      "graph": {
        "stages": [
          {
            "id": "separate",
            "kind": "audio_separation",
            "required": true,
            "component_ids": ["separation.demucs"],
            "companion_ids": ["demucs"],
            "depends_on": [],
            "inputs": ["source.audio.mix"],
            "outputs": ["artifact.stem.guitar"]
          },
          {
            "id": "assemble",
            "kind": "chart_assembly",
            "instrument": "guitar",
            "required": true,
            "component_ids": [],
            "companion_ids": [],
            "depends_on": ["separate"],
            "inputs": ["artifact.stem.guitar"],
            "outputs": ["chart.guitar.expert"]
          }
        ],
        "outputs": [
          {
            "instrument": "guitar",
            "stage_id": "assemble",
            "artifact_id": "chart.guitar.expert",
            "difficulty": "Expert"
          }
        ]
      }
    }
  }
}
```

Profile validation hashes every required graph component and returns the safe
companion/version requirements. Chart preflight resolves every graph stage and
terminal output. If no handler exists for that exact profile capability, it
reports `execution: not_available` and required stages as `unavailable`; it
does not invoke the legacy batch pipeline or claim a deployable profile.

The first chart execution capability is intentionally narrow:
`guitar.hybrid-v2-rule/v1`. It requires a bundle-verified onset checkpoint and
model-config fingerprint, a typed profile configuration, and an installed
Basic Pitch runtime. Basic Pitch's TensorFlow dependency currently requires a
managed Python 3.11 environment; `strum-worker probe --json` reports whether
it is available. Its request references an already-validated preflight request
plus private input/output locations:

```json
{
  "preflight_request": "/private/guitar-preflight.json",
  "audio_path": "/private/guitar-stem.ogg",
  "output_dir": "/private/chart-run"
}
```

Run it with `strum-worker chart run --request /path/to/owned-chart-run.json
--json`. STRUM writes Expert-only `notes.mid` and `run.json`, recording profile
and component hashes. It never enables the learned fret mapper, uses no
`STRUM_GUITAR_*` overrides, and cannot make Hard/Medium/Easy charts unless a
separate explicit STRUM difficulty profile is selected.

`drums.v14-expert/v1` is also executable. It accepts exactly one
bundle-verified, tensor-only V14 8-class checkpoint with fixed V14
preprocessing and emits Expert Drums with direct class thresholds. It
intentionally does **not** run legacy postprocessing, ensemble, cymbal, or
multiclass fallback stages. The legacy multi-instrument batch pipeline remains
outside this worker execution contract because its companions and fallbacks are
not yet declared profiles.

```bash
python -m src.song_source_catalog /path/to/catalog
python scripts/build_guitar_catalog_manifest.py /path/to/catalog \
  --output /path/to/views/guitar.json
python scripts/preprocess_guitar_windows.py \
  --manifest /path/to/views/guitar.json \
  --catalog-root /path/to/catalog --cache-dir /path/to/cache
```

The current Guitar onset/fret adapter requires Expert Guitar coverage and
prefers `audio.guitar`, falling back to `audio.mix`. New task builders use
`load_catalog()` and `select_training_sources()` rather than scanning source
folders.

### Drums task manifest

The Drums onset/classifier pipeline has the same catalog-only boundary.  The
task view selects Expert Drums and prefers `audio.drums`, falling back to
`audio.mix`. STRUM derives its 8-lane pro-drums labels from the managed
`notes.mid` in memory, so OCTAVE does not need to generate legacy
`drums_labels.json` files.

```bash
python scripts/build_drums_catalog_manifest.py /path/to/catalog \
  --output /path/to/views/drums.json
python scripts/preprocess_onset_windows.py \
  --manifest /path/to/views/drums.json \
  --catalog-root /path/to/catalog \
  --output-dir /path/to/drums-cache --split both
```

The generated cache index records a path-free lineage contract: catalog ID and
content hash, pipeline ID/version, task-view hash, deterministic split policy,
selected source IDs and asset hashes, plus a preprocessing configuration hash.
Use this lineage when deciding whether a drum checkpoint may be resumed,
fine-tuned, or deployed for auto-charting. The legacy folder-based manifest is
still supported for historical data, but must not be used for OCTAVE catalogs.

### Remaining instrument and derived-label task views

`build_catalog_task_manifest.py` is STRUM's shared adapter for catalog-backed
training data. It supports `bass`, `keys`, `vocals`, `vocals_activity`,
`vocals_phrase_boundaries`, `pro_guitar`, `pro_bass`, `pro_keys`,
`fret_mapper_guitar`, `fret_mapper_bass`, `section_guitar`, and `section_bass`.
Every view records the versioned pipeline ID, catalog control
fingerprint, source IDs and input hashes, deterministic split algorithm/seed,
and a fingerprint of portable preprocessing settings. It also declares the
immutable label-source schema and the exact approved MIDI track names selected
for each song (for example `PART VOCALS`, `PART REAL_GUITAR`, or
`PART REAL_KEYS_X`). A future trainer therefore receives explicit event
semantics rather than inferring track conventions from its own source tree.
STRUM revalidates both declarations against the catalog before resolving the
ephemeral managed paths. The view never records an OCTAVE source path.

Pro tasks additionally pass through `src.pro_target_manifest` during worker
Prepare. The resulting `strum-pro-target-task-manifest/v1` embeds only the
immutable catalog task view plus decoded label events. At train time the
decoder revalidates catalog lineage and re-derives every target from the
managed MIDI asset, so edited target JSON, drifted MIDI, invalid note pairing,
unsupported Pro technique channels, or a standard-track fret above 17 cannot
silently become training data. This closes the label-decoding gap only; it does
not create a Pro trainer or auto-chart profile.

```bash
# Any chart/audio family: only Expert coverage and allowed catalog records.
python scripts/build_catalog_task_manifest.py /path/to/catalog \
  --task bass --output /path/to/views/bass-v1.json \
  --preprocessing-json '{"sample_rate":22050,"window_seconds":5}'

# Derived labels are also catalog-backed. Paths are resolved only while the
# builder/preprocessor is running, and the label file stores source IDs.
python scripts/build_catalog_task_manifest.py /path/to/catalog \
  --task section_guitar --output /path/to/views/section-guitar-v1.json
python scripts/build_catalog_section_labels.py \
  --manifest /path/to/views/section-guitar-v1.json \
  --catalog-root /path/to/catalog --out /path/to/views/section-labels-v1.json
python scripts/preprocess_section_windows.py \
  --labels /path/to/views/section-labels-v1.json \
  --catalog-manifest /path/to/views/section-guitar-v1.json \
  --catalog-root /path/to/catalog --cache-dir /path/to/cache

# The mapper accepts a mapper task view directly. It persists catalog source
# IDs in its cache, not paths from imported packages.
python scripts/build_mapper_dataset.py \
  --catalog-manifest /path/to/views/fret-mapper-guitar-v1.json \
  --catalog-root /path/to/catalog --cache-dir /path/to/mapper-cache
```

These task views make each family catalog-ready. Guitar/Bass section task views
also feed a bounded worker experiment: STRUM derives six chart-pattern labels,
materializes catalog-split log-mel windows, and trains `SectionClassifier/v1`.
The resulting `section_classifier.guitar` or `section_classifier.bass`
component has no inference profile and is not an auto-chart model. Its worker
feature extractor is explicitly `section-logmel-torchaudio-windows/v1`, while
the legacy `SectionRouter` currently computes a different declared librosa
frontend. Similar mel dimensions are not evidence of compatibility, so STRUM
will not package a worker checkpoint into that router or let it replace the
router's checkpoint by path. Before a section component can affect a chart, it
needs: an exact feature-extractor contract, a tensor-only bundle/profile
loader, held-out section calibration, a held-out router-on/off chart-impact
ablation, and composition with an executable instrument-specific chart
profile. Guitar and Bass require separate composition contracts.
Keys and Pro-instrument generic task views remain source contracts, not a
claim that their future models share the five-lane representation. The generic
Vocal descriptor is also explicitly planned: its structured contract names the
exact lead track, already-trainable activity/pitch and phrase components, and
the lyric, talky, harmony, composition, evaluation, packaging, and execution
stages still required for a selectable profile.
### Catalog-backed chart-transform tasks

The `chart_transform.five_lane/v1` pipeline learns Expert → Hard, Medium, or
Easy chart pairs for Guitar, Bass, Keys, or Drums. It consumes only `allowed`
catalog records containing both Expert and the requested target difficulty;
OCTAVE remains the importer and curation boundary.

```bash
python scripts/prepare_catalog_chart_pairs.py \
  --catalog-root /run/media/ash/portable-ai/strum/catalogs/octave-curated-catalog \
  --output-dir /run/media/ash/portable-ai/strum/tasks/guitar-expert-hard-v1 \
  --instrument guitar \
  --target-difficulty Hard
python scripts/train_chart_transform.py \
  --config /path/to/chart-transform.yaml \
  --dataset-manifest /run/media/ash/portable-ai/strum/tasks/guitar-expert-hard-v1/dataset-manifest.json
```

`--describe-pipeline` prints the stable pipeline descriptor for an OCTAVE
worker. Each task view records its pipeline ID/version, catalog manifest and
records hashes, source IDs with `notes.mid` hashes, deterministic source-ID
split assignments, and preprocessing configuration hash. Training revalidates
that lineage and preserves it in `training-metadata.json`; neither artifact
contains original package locations or raw local paths.

`prepare_instrument_chart_pairs.py` remains a local standalone bridge for
non-catalog experiments. It is not an OCTAVE integration input and must not be
used by an OCTAVE worker to scan source folders.

## Development

Developed on NVIDIA DGX Spark (GB10 GPU, CUDA 12.8). Trained on ~5,000 human-authored pro drum charts from the Clone Hero community.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Technical specification
- [Roadmap](docs/ROADMAP.md) — Development milestones

## Acknowledgments

- [Demucs](https://github.com/adefossez/demucs) — Audio source separation
- [OpenAI Whisper](https://github.com/openai/whisper) — Speech recognition
- [librosa](https://librosa.org/) — Audio analysis
- Clone Hero / YARG communities — Chart format documentation

## License

MIT
