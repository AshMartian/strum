# STRUM Architecture

This document describes the **as-built** STRUM research pipeline and the
separate, versioned worker boundary used by OCTAVE. For high-level usage see
[README.md](../README.md); for status & next steps see
[ROADMAP.md](ROADMAP.md).

## Deployment status and ownership

The diagram below is the legacy multi-instrument batch architecture. It is not
a claim that every stage is a deployable `strum-worker` profile. The worker
fails closed to the profiles declared by a validated model bundle:

| Capability | Worker status | Output boundary |
| --- | --- | --- |
| `guitar.hybrid-v2-rule/v1` | Executable | Expert Guitar only |
| `drums.v14-expert/v1` | Executable | Expert Drums only, direct V14 (no legacy ensemble/fallbacks) |
| `difficulty.transform/v1` | Executable | Learned five-lane Expert → Hard/Medium/Easy transform for Guitar, Bass, Keys, or Drums |
| Guitar onset/fret | Worker-trainable, experiment-only | Safe task views plus path-free experiment bundles; auto-chart profile packaging remains required |
| Bass onset/fret | Worker-trainable, experiment-only | Revalidated `PART BASS` task views and distinct `bass.onset`/`bass.fret` components; a Bass evaluator/profile is required before auto-charting |
| `drums.onset-classifier-evaluation/v1` | Executable evaluator only | Prepared V2 onset windows → eight class probabilities; explicitly not a chart handler |
| Pro Guitar, Pro Bass, Pro Keys | Catalog-ready task views, training planned | No worker trainer or deployable profile; exact REAL_* labels and structured missing stages are declared |
| Vocals, Guitar/Bass mapper, Guitar/Bass section | Worker-trainable experiments | No deployable profile without their dedicated evaluation/package/runtime gates |
| Legacy batch pipeline | Research / compatibility scripts | Not a worker execution handler |

OCTAVE owns source import, rights decisions, curation, runtime selection, and
job/process lifecycle. STRUM owns preprocessing, labels, learned models,
checkpoint/profile compatibility, and difficulty semantics. In particular,
OCTAVE must not implement a deterministic Expert-to-lower-difficulty mapper:
an Expert worker output stays Expert-only until a declared STRUM learned
difficulty profile is run.

### Drums classifier evaluation package boundary

`strum-worker checkpoint package --request …` may package a completed
catalog-worker `drums.onset-classifier/v1` experiment only after it verifies
the experiment ledger, original configuration hash, checkpoint hash/length,
and the exact V2 architecture/preprocessing contract. The portable bundle
contains a Stage-2 `OnsetClassifier/v2` evaluator. It takes only prepared
fine/coarse mel windows plus the 64-value onset context and returns eight
sigmoid class probabilities. It has no audio onset detector, velocity head, or
MIDI writer, so chart preflight reports `execution: not_available` for it.

The direct `drums.v14-expert/v1` profile remains the only executable Expert
Drums chart path. A future replacement must package and evaluate a compatible
onset detector, classifier/velocity semantics, preprocessing, calibration,
and chart output contract together; a V2 classifier checkpoint alone is not
such a profile.

## 1. Legacy system overview

STRUM converts a single audio file (`song.ogg/.wav/.mp3`) into a Clone Hero /
YARG chart package containing PART DRUMS, PART GUITAR, PART BASS, PART VOCALS
(with lyrics), and PART KEYS — all aligned to a common tempo grid.

```
                           audio (44.1 kHz)
                                 │
                       ┌─────────▼──────────┐
                       │  Demucs htdemucs   │
                       │  6-stem separation │
                       └─────────┬──────────┘
                                 │
       ┌────────────┬────────────┼────────────┬────────────┐
       │            │            │            │            │
   drums.wav    guitar.wav    bass.wav    vocals.wav    other.wav
       │            │            │            │            │
       ▼            ▼            ▼            ▼            ▼
   Two-stage     Hybrid       Hybrid       Whisper      Spectral
   CRNN +        onset +      onset +      + pYIN +     keyboard
   ensemble      Basic Pitch  Basic Pitch  alignment    detector
   classifier    + fret map   + fret map   + LRCLIB     + piptrack
       │            │            │            │            │
       └────────────┴────────┬───┴────────────┴────────────┘
                             │
                ┌────────────▼─────────────┐
                │  BPM grid alignment      │
                │  (±5 BPM @ 0.1 res +     │
                │  beat-zero phase snap)   │
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │  Phase shift + 32nd-note │
                │  snap (per-lane roll     │
                │  detection)              │
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │ Legacy difficulty stage  │
                │ (not OCTAVE deployment) │
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │  notes.mid + song.ini    │
                │  + album art + audio     │
                └──────────────────────────┘
```

## 2. Drums Pipeline

The flagship pipeline — a two-stage neural system with audio-coupled rescue
passes and ensemble voting. Implemented in `scripts/batch_infer_hybrid.py` and
called from `batch_pipeline.py`.

### 2.1 Stage 1 — Onset Detection (`TwoStageDrumsCRNN`, V14)

Mel spectrograms (128 mel bins, 22050 Hz, hop 512) → CNN → BiLSTM → 1-D
onset probability per frame. Detected onsets become **lane-agnostic** drum hits
with timestamps. Trained on ~5k human pro-drum charts; verified F1 = 93.9%.

Architecture: `src/models/drums_v13.py::TwoStageDrumsCRNN` (V14 reuses the V13
class with a stronger checkpoint).

### 2.2 Stage 2 — Lane Classification (Ensemble)

Each detected onset is classified across 8 lanes (Kick, Snare, Hi-Hat, Crash,
Ride, High Tom, Mid Tom, Floor Tom) by an ensemble of independently-trained
`OnsetClassifier` models with `PER_CLASS_WEIGHTS`:

| Member | Config | Notes |
|--------|--------|-------|
| V2  | `onset_classifier.yaml`            | Original baseline, full mix |
| V6  | `onset_classifier_v6.yaml`         | Stronger snare/hi-hat head |
| V12c| `onset_classifier_v12_clean.yaml`  | Trained on clean Demucs stems |
| V15 | `onset_classifier_v15.yaml`        | Bg-mel subtraction |
| V16 | `onset_classifier_v16.yaml`        | Cymbal-aware loss |

Voting uses class-weighted soft-max averaging followed by argmax. Verified
ensemble F1 = 85.2%; best single member (V12c) = 83.8%.

### 2.3 Rescue & Refinement Passes

After classification, six audio-coupled passes index back into the mel-frame
probability tensor `mc_frame_probs` to fix systematic errors. **Critically, all
six must run on the original time base — before any phase shift is applied**
(see §6).

| Pass | Purpose |
|------|---------|
| `phase3_onset_rescue` | Recover missed quiet onsets via mel-frame peak picking |
| `phase3_cymbal_cooccurrence_rescue` | Add missing cymbals when kick+cymbal frame coincides |
| `apply_cymbal_to_tom_rescue` | Convert false-positive cymbals to toms by spectral centroid |
| `apply_drumsep_hits_arbiter` | Cross-check against Demucs drum-separation stems |
| `spectral_reclassify` | Resolve tom-vs-cymbal confusion by harmonic ratio |
| `apply_tom_refinement_filter` | Tom-refinement CNN (`src/models/tom_refinement.py`) |

### 2.4 Post-Processing (`scripts/chart_postprocess.py`)

* Bidirectional iterative streak smoothing (collapses isolated mis-classifications
  inside a uniform streak)
* `kick_suppresses_floor_tom` (kick + floor tom on same frame → drop the tom)
* `enforce_single_tom_per_onset` (one tom marker per onset, max 110/111/112)
* `cap_close_hands` (no two cymbals < 60 ms apart unless tom-roll context)
* `protect_tom_fills` (preserve dense tom rolls from over-aggressive snap)
* Lane conflict resolution + velocity normalization

## 3. Guitar & Bass Pipeline

Both guitar and bass share the same hybrid architecture
(`src/inference/guitar_hybrid_v2.py`). Backend selection via env vars:

| Backend | Onset source | Pitch source | Use case |
|---------|--------------|--------------|----------|
| `hybrid` *(default)* | V2 onset CRNN on Demucs stem | Spotify Basic Pitch | Best balance |
| `neural` | V2 onset CRNN on full mix | V2 fret head | Pure-neural baseline |
| `basicpitch` | Basic Pitch onsets | Basic Pitch | Fallback for sparse onset stems |
| `rule` | librosa onset_detect | pYIN | Legacy, no neural component |

### 3.1 Onset Detection (V1/V2 Guitar CRNN)

`src/models/guitar_v1.py::OnsetCRNN` — single-output onset head trained on
isolated `guitar.ogg` stems from ~5k charts. V2 adds a deeper backbone and
section-aware peak-thresholding. Configs: `guitar_v1.yaml`, `guitar_v2.yaml`.

### 3.2 Polyphonic Pitch (Basic Pitch)

[Spotify Basic Pitch](https://github.com/spotify/basic-pitch) transcribes each
detected onset window into 0–N MIDI pitches (chord support). For bass we
override the model's MIDI range to 24–67 via `STRUM_BP_MIN_PITCH` /
`STRUM_BP_MAX_PITCH`, and use a softer onset peak threshold
(`STRUM_GUITAR_PEAK_THR=0.35`) because bass attacks are weaker than guitar.

### 3.3 Pitch → Fret Mapping

Default: **rule-based register allocation**. Pitches are bucketed into the
five Clone Hero frets by absolute MIDI value, with chord-shape preservation.
HOPO threshold is 170 ms.

Optional: **`PitchToFretMapper` (V4)** — a learned mapper trained on ~5k chart
pitch→fret pairs (`scripts/build_mapper_dataset.py` +
`scripts/train_fret_mapper.py`). Enabled with `STRUM_FRET_MAPPER=1`.

### 3.4 Section Router

`src/inference/section_router.py` predicts per-1-second section labels
(verse/chorus/solo/etc.) using `src/models/section_classifier.py` and modulates
the onset peak threshold per section, preventing over-charting in quiet verses
and under-charting in dense choruses. Optional, gated by checkpoint presence.

## 4. Vocals Pipeline (`scripts/vocals_charter.py`)

1. **Lyric transcription** — OpenAI Whisper extracts word-level timestamps
   from the Demucs vocal stem.
2. **Pitch contour** — `librosa.pyin` tracks vocal F0 at 10 ms resolution.
3. **Word ↔ pitch alignment** — Whisper word boundaries are warped against
   pitch onsets via a dynamic-programming alignment so each word lands on the
   pitch attack instead of the model's centroid estimate.
4. **Lyrics fetching** — Optional synced lyrics from
   [LRCLIB](https://lrclib.net/) and [Lyrics.ovh](https://lyrics.ovh/).
5. **Harmony detection** — Configurable presence threshold (default 30%); a
   pitch line is added to PART HARM2/HARM3 only if it appears in ≥30% of
   vocal sections.

Output dataclasses (`VocalNote`, `VocalPhrase`) use **seconds** for timestamps,
not ms — see §6.5.

## 5. Keys Pipeline (`scripts/keys_charter.py`)

1. **Keyboard detection** — Spectral flatness + harmonic ratio analysis on
   the Demucs `other` stem identifies regions where a keyboard is active.
2. **Onset detection** — `librosa.onset_detect` over the keyboard-active
   regions only.
3. **Pitch extraction** — `librosa.piptrack` per onset window, then Basic
   Pitch refinement when `STRUM_KEYS_BACKEND=basicpitch`.
4. **Dual output** — Both 5-lane simplified PART KEYS and full-range PART
   REAL_KEYS (Pro Keys) are written.

## 6. Tempo Detection & Cross-Instrument Grid Alignment

The single most important system-wide invariant.

### 6.1 BPM Refinement

Initial BPM from `librosa.beat.beat_track`, then a ±5 BPM grid search at
0.1 BPM resolution. For each candidate BPM we measure phase coherence using
**circular statistics** on `(onset_time mod beat_period)`, picking the BPM
with maximum unit-vector magnitude. Reduces grid error from ~175 ms to
~35 ms on typical tracks.

### 6.2 Phase Offset

Once the BPM is locked, `phase_offset_ms` is the time of the **first
detected onset** modulo the beat period. This is more robust than the
circular mean of all onsets, which drifts when the grid-aligned BPM differs
from the librosa estimate.

### 6.3 The Critical Ordering

Every transcriber emits events on the **raw audio time base** so that the six
drum rescue passes (§2.3) can index `mc_frame_probs[time_ms * sr / hop]`
correctly. The phase shift and grid snap happen **after** all rescue passes
have run, in `transcribe_drums()` and the cross-instrument loop in
`batch_pipeline.py`.

```python
for chart in (drums, guitar.notes, guitar.chords,
              bass.notes, bass.chords,
              keys, vocal_phrases, vocal_notes):
    for ev in chart:
        ev.time += phase_offset           # ms or seconds per dataclass
        ev.time = snap_to_grid(ev.time, grid_32nd, roll_window=grid_ms*1.1)
```

### 6.4 Snap-to-Grid with Roll Detection

A naive snap collapses fast double-strokes and tom rolls. The snap function
checks for a same-(lane, is_cymbal) neighbor within
`_roll_window_ms = grid_ms * 1.1` and skips the snap if one exists, preserving
rolls.

### 6.5 Time-Base Inventory

Different transcribers use different time units. The cross-instrument loop
respects this:

| Dataclass | Time field | Unit |
|-----------|------------|------|
| `DrumHit` | `time_ms` | milliseconds |
| `GuitarNote`, `GuitarChord` | `time_ms` | milliseconds |
| `KeysNote` | `time_ms` | milliseconds |
| `VocalNote`, `VocalPhrase` | `start_time`, `end_time` | **seconds** |

## 7. Chart Export

### 7.1 MIDI (`src/export/midi.py`)

Standard MIDI File Type 1, 480 ticks per quarter note. Tracks:

* Track 0 — Tempo map + time signatures
* Track 1 — Section markers
* `PART DRUMS` — Pro drums (lanes 96–100, tom markers 110–112)
* `PART GUITAR` / `PART BASS` — 5-fret (96–100), HOPO/tap modifiers
* `PART VOCALS` — Pitched phrases + lyric meta-events
* `PART KEYS` / `PART REAL_KEYS_X` — 5-lane + Pro Keys

### 7.2 Legacy difficulty generation (`scripts/chart_enhancer.py`)

`chart_enhancer.py` is legacy batch behavior, not an OCTAVE-owned deployment
feature and not a generic worker fallback. The table documents its historical
heuristic configuration only. Worker lower-difficulty output requires the
explicit `difficulty.transform/v1` learned STRUM profile and records its model
provenance in the run manifest.

| Difficulty | Notes/sec cap | Max chord size | Notes |
|------------|---------------|----------------|-------|
| Expert | 12 | 4 | Full chart |
| Hard   | 9  | 3 | Drop ghost notes |
| Medium | 6  | 2 | Simplify rolls |
| Easy   | 4  | 1 | Downbeats only |

### 7.3 song.ini

Generated with title, artist, charter (`STRUM`), genre, year, BPM, and
per-instrument difficulty ratings. Cover art is fetched from the audio file's
embedded metadata or via the iTunes search API.

## 8. Module Map

```
src/
├── models/
│   ├── drums_v13.py              # TwoStageDrumsCRNN (V14 ckpt)
│   ├── drums_v14_dataset.py      # Drum onset dataset (bg-mel subtraction)
│   ├── onset_classifier.py       # 8-lane drum classifier (ensemble member)
│   ├── onset_classifier_dataset.py, *_cached_dataset.py
│   ├── tom_refinement.py         # Tom-vs-cymbal CNN
│   ├── guitar_v1.py              # Guitar onset CRNN (V1/V2)
│   ├── section_classifier.py     # Per-1s section labeler
│   ├── bg_mel.py                 # Background-mel subtraction utilities
│   └── common.py
├── inference/
│   ├── guitar_hybrid_v2.py       # Hybrid Guitar/Bass research backend
│   ├── guitar_neural.py          # Neural-only backend
│   ├── guitar_bass.py            # Dataclasses + rule backend
│   ├── section_router.py         # Section-aware onset gating
│   └── c3_rules.py               # Clone Hero charting conventions
├── preprocessing/
│   ├── parsers/                  # .mid + .chart parsers
│   ├── alignment.py              # Audio-chart alignment (training)
│   └── separation.py             # Demucs wrapper
├── export/
│   ├── midi.py                   # MIDI writer (all parts)
│   └── chart.py                  # .chart format writer
└── lyrics/                       # LRCLIB + Lyrics.ovh fetcher
```

## 9. Legacy training inventory

| Model | Trainer | Preprocess | Config |
|-------|---------|------------|--------|
| Drum onset CRNN (V14) | `train_onset_classifier.py` | `preprocess_onset_windows.py` | `drums_v14.yaml` |
| Drum classifier ensemble | `train_onset_classifier.py` | `preprocess_onset_windows.py` | `onset_classifier_*.yaml` |
| Tom-refinement CNN | `train_tom_refinement.py` | (uses Demucs drum stem) | inline |
| Five-lane Guitar/Bass onset + fret CRNN (V1) | `train_guitar_v1.py` | revalidated Guitar/Bass catalog task view → `preprocess_guitar_windows.py` | `guitar_v1.yaml` |
| Pitch→fret mapper (V4) | worker-owned `strum.fret-mapper/{guitar,bass}/v1` → `train_fret_mapper.py` | catalog task view → `build_mapper_dataset.py` | worker bundle config |
| Section classifier | `train_section_classifier.py` | `build_section_labels.py` → `preprocess_section_windows.py` | inline |

All trainers log to W&B (`WANDB_MODE=offline` to disable). The Guitar/Bass
fret-mapper worker is a narrow exception: it invokes the established builder
and MLP trainer only after catalog revalidation, persists the task-view split
in each derived cache file, and packages a hash-verified component with no
inference profile. It requires the `pitch` optional dependency. Other legacy
script interfaces remain preparation-only; catalog readiness does not make a
trainer worker-runnable, and any script output is not deployment-ready until
STRUM evaluates and packages a model bundle/profile.

## 10. Catalog and worker contract

OCTAVE writes an `octave-song-source-catalog/v1` of already-authorized managed
assets. STRUM validates it, selects only `training_use: allowed` records, and
creates path-free task views. STRUM never imports `.sng`, `.rb3con`, ZIP, or
original source folders, and task/experiment/checkpoint manifests must not
contain those locations.

Each discovered pipeline reports independent `preparation_status`,
`training_status`, and stable `training_requirements`. This makes a catalog
task view useful without overstating worker training or deployment support.
The Pro descriptors also expose a `strum-planned-training-contract/v1`: exact
REAL_* source-track identities, required Expert label semantics, ordered
missing stages, and an explicit `execution.status: not_available`. Pro Keys
accepts only `PART REAL_KEYS_X` for the Expert path; Pro Guitar/Bass retain
the standard versus `_22` source variant for their future encoder. OCTAVE can
render those facts directly rather than treating a planned descriptor as a
disabled version of a five-lane model. Section descriptors now expose their
real catalog worker while retaining evaluation, profile-packaging, and runtime
integration requirements.

Discovery is dynamic rather than hard-coded in OCTAVE:

```bash
strum-worker probe --json
strum-worker pipeline list --json
strum-worker catalog inspect --catalog-root /private/catalog --pipeline guitar.onset-fret/v1 --json
strum-worker catalog inspect --catalog-root /private/catalog --pipeline bass.onset-fret/v1 --json
strum-worker catalog inspect --catalog-root /private/catalog --pipeline keys.onset-fret/v1 --json
```

OCTAVE renders only the selected descriptor's `prepare_schema` and
`train_schema`. It uses `dataset prepare --json-events` and `train start
--json-events` for supervised work; event lines have a request-derived opaque
job ID, monotonic sequence, stage, progress, state, and safe code/message.
OCTAVE owns cancellation by terminating the supervised process group, rather
than requiring STRUM to retain a private-path job. A model folder is validated
separately from a STRUM runtime with `checkpoint inspect`, `inference profile
validate`, `chart preflight`, and `chart run`.

### Guitar V1 deployment gate

`guitar.onset-fret/v1` produces an experiment bundle, not an auto-chart
profile. To deploy a newly trained pair, STRUM first revalidates the selected
catalog task view and measures its validation split, then copies the verified
experiment into a separate immutable profile bundle:

```bash
strum-worker guitar profile evaluate \
  --bundle-root /private/experiment/bundle \
  --task-view /private/task-view.json \
  --catalog-root /private/catalog \
  --output /private/evaluation.json --device cuda

strum-worker guitar profile package \
  --experiment /private/experiment \
  --evaluation /private/evaluation.json \
  --output /private/deployable-guitar-v1 \
  --profile guitar-v1-expert \
  --minimum-onset-f1 0.50 --minimum-fret-f1 0.50
```

Packaging copies, rather than mutates, the experiment; freezes the exact
22.05 kHz / 128-mel configuration and inference thresholds; and supports only
Expert Guitar. `chart run` loads only tensor state dictionaries whose component
and evaluation hashes were preflighted. Older Guitar experiments without the
portable `strum-guitar-neural-model-config/v1` component configuration are
intentionally not packageable and should be retrained.

### Bass V1 experiment gate

`bass.onset-fret/v1` deliberately reuses only the five-lane CRNN *implementation*,
not the Guitar task or profile contract. Its preparation task kind is
`bass_onset_fret`, whose catalog-validated label schema selects `PART BASS` and
the standard Expert lanes 96--100. The worker invokes the shared feature
extractor with `--instrument bass`, then produces only `bass.onset` and
`bass.fret` components with a `strum-bass-neural-model-config/v1` configuration.

The raw experiment has
`deployment_status: requires_bass_profile_evaluation_and_packaging` and cannot
be selected by `chart run` or packaged as `guitar.neural-v1-expert/v1`.
`strum-worker bass profile evaluate` resolves the approved catalog only while
it validates the held-out task-view split against `PART BASS`; it records only
aggregate metrics and content hashes. `strum-worker bass profile package` then
copies a gate-satisfying experiment into an immutable
`bass.neural-v1-expert/v1` bundle. The typed runtime consumes only
`bass.onset`/`bass.fret`, tensor-only checkpoints, and that profile's
configuration, writes `PART BASS`, and supports only Expert output.

### Keys V1 experiment gate

`keys.onset-fret/v1` likewise reuses only the five-lane CRNN implementation,
not a Guitar or Bass task/profile contract. Its `keys_onset_fret` task view
revalidates the catalog, selects only `PART KEYS`, and uses the standard Expert
five-lane encoding (96--100). The worker invokes the shared feature extractor
with `--instrument keys`, then emits only `keys.onset` and `keys.fret`
components with `strum-keys-neural-model-config/v1` configuration.

The experiment remains
`deployment_status: requires_keys_profile_evaluation_and_packaging` and has no
`inference_capability`. It is not a replacement for the legacy Keys charter;
it needs Keys-specific held-out evaluation and a runtime/profile contract
before a chart handler may select it.

### Vocal activity/pitch experiment gate

`vocals.note-activity/v1` is a catalog worker experiment, not a conversion of
vocal charts into five-lane data. Its `vocals_activity` task view is
revalidated against the selected catalog and may derive labels only from
`PART VOCALS`. The preprocessor produces 22.05 kHz / 128-mel windows with two
frame targets: pitched lead-vocal activity and MIDI pitch 36--84. It retains
observed phrase-marker and lyric-event counts only as metadata.

The resulting `vocals.frame_activity_pitch` bundle has no profile and reports
`deployment_status: requires_vocals_profile_evaluation_and_packaging`. It
cannot be selected by `chart run` or used to invent phrase, lyric, talky, or
harmony output; those require separately evaluated components and a complete
Vocal-specific profile contract.

## 11. Hardware

Developed on NVIDIA DGX Spark (GB10 GPU, CUDA 12.8). Inference runs in
~real-time on a single 12 GB GPU; training one drum classifier takes
~6–8 hours on the same hardware.
