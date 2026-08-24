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
| Guitar onset/fret | Worker-trainable; evaluated profile executable | Safe task views plus path-free experiment bundles; only an evaluated `guitar.neural-v1-expert/v1` profile writes Expert Guitar |
| Bass onset/fret | Worker-trainable; evaluated profile executable | Revalidated `PART BASS` task views and distinct `bass.onset`/`bass.fret` components; only an evaluated `bass.neural-v1-expert/v1` profile writes Expert Bass |
| Keys onset/fret | Worker-trainable; evaluated profile executable | Revalidated `PART KEYS` task views and distinct `keys.onset`/`keys.fret` components; only an evaluated `keys.neural-v1-expert/v1` profile writes Expert Keys |
| `drums.onset-classifier-evaluation/v1` | Executable evaluator only | Prepared V2 onset windows → eight class probabilities; explicitly not a chart handler |
| Pro Guitar, Pro Bass, Pro Keys | Catalog-ready exact-REAL_* known-event and audio-proposal candidates | `candidate_kind` selects one raw component/preprocessing contract; no sequence decoder, profile, or deployable chart handler |
| Vocals, Guitar/Bass mapper, Guitar/Bass section | Worker-trainable experiments | Vocal activity/pitch, phrase-boundary, lyric, and lead talky components are distinct and non-deployable; no profile without remaining composition/evaluation/runtime gates |
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
`scripts/train_fret_mapper.py`). The legacy research bridge is explicitly
enabled with `STRUM_GUITAR_FRET_MAPPER=learned`; it is not a bundle profile and
fails closed when unavailable. Catalog-worker mapper experiments record the
Basic Pitch distribution/version and cannot be promoted until a typed profile
pins a tensor-only decoder, Basic Pitch/Viterbi policy, compatible onset
source, and end-to-end held-out chart evaluation.

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
| Section classifier | worker-owned `strum.section-classifier/{guitar,bass}/v1` → `train_section_classifier.py` | revalidated exact `PART GUITAR`/`PART BASS` task track → `build_catalog_section_labels.py` → `preprocess_section_windows.py` | experiment-only bundle with declared `section-logmel-librosa-router-windows/v1`; exact legacy router frontend, but no router profile |

All trainers log to W&B (`WANDB_MODE=offline` to disable). The Guitar/Bass
fret-mapper worker is a narrow exception: it invokes the established builder
and MLP trainer only after catalog revalidation, persists the task-view split
in each derived cache file, and packages a hash-verified component with no
inference profile. It requires the `pitch` optional dependency. Other legacy
script interfaces remain preparation-only; catalog readiness does not make a
trainer worker-runnable, and any script output is not deployment-ready until
STRUM evaluates and packages a model bundle/profile.

Catalog fret-mapper components now have an equally narrow typed candidate
loader. `strum-fret-mapper-weights/v1` contains only tensor state and tensor
normalization values (plus primitive architecture metadata), and
`load_fret_mapper_candidate` uses `torch.load(weights_only=True)`, checks the
declared `FretMapperMLP/v1` dimensions, and loads state strictly. This is a
checkpoint safety/evaluation boundary only: it is not a Guitar/Bass chart
profile, has no worker execution capability, and does not bridge the Basic
Pitch/Viterbi/held-out chart-evaluation requirements.

Section training is also intentionally experiment-only. The catalog worker and
the established `SectionRouter` now import one exact librosa frontend contract
(decode/resampling, Slaney Mel, constant padding, full-song frame slicing, and
per-window normalization). `strum-worker section profile evaluate` validates
the declared `SectionClassifier/v1` state with `torch.load(weights_only=True)`,
selects a temperature on validation windows, and writes only test-split
metrics. Candidate config, experiment, and report must share one immutable
task-view digest; package validation also recomputes report metrics from the
confusion matrix and aggregate calibration evidence. The corresponding
`section.classifier-evaluation/v1` package has `evaluation_only` difficulty
scope and no chart handler. The runtime must still not substitute it into the
router by filename. A future composed chart profile must pass a held-out
router-on/off chart-impact ablation and bind a registered instrument-specific
execution handler before it can change chart output.
If a catalog task contains a test split, the trainer reports best-validation
checkpoint test metrics in the experiment. That report is not a profile
promotion decision or a substitute for the router-on/off chart-impact gate.

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
missing stages, and an explicit `execution.status: not_available`. Their two
available worker experiments are the known-reference-event candidate
(`strum-pro-known-reference-event-window/v1`) and the experimental,
free-running audio proposal candidate (`strum-pro-arbitrary-audio-window/v1`).
The proposal candidate scores bounded offline audio windows without MIDI at
inference. Its deterministic negative policy excludes every center whose
asymmetric feature window would contain a REAL event onset; it produces only
event-proposal scores, not attributes, a sequence decoder, MIDI, a profile,
or a chart. The path-free
`strum-candidate-checkpoint-output-contracts/v1` descriptor field binds the
selected `candidate_kind` to exactly one component identity and preprocessing
contract: known-event attributes use `pro-logmel-event-windows/v1`, while the
proposal uses `pro-logmel-event-proposal-windows/v1`. Static
`checkpoint_outputs` is intentionally empty for Pro because the candidates
are mutually exclusive; every map entry remains raw-experiment-only with no
profile or chart execution. Pro Keys
accepts only `PART REAL_KEYS_X` for the Expert path; Pro Guitar/Bass retain
the standard versus `_22` source variant in a worker-produced,
`strum-pro-target-task-manifest/v1` target view. STRUM decodes those immutable
catalog assets into exact string/fret/technique or Pro Keys pitch/range-shift
events and re-derives them before a future trainer may consume the view. Its
`pro-logmel-event-windows/v1` cache revalidates audio and whole-song tempo
maps, retaining target-track variants and event semantics rather than passing
five-lane labels to a generic feature builder. It remains research-only: no
Pro execution/profile is discoverable. Section descriptors now expose their
real catalog worker while retaining evaluation, profile-packaging, and runtime
integration requirements.

The learned five-lane difficulty transform can be prepared as chart-only or
with the bounded `rms_onset_v1` catalog-audio baseline. Audio conditioning is
a task-view choice: STRUM retains only approved audio roles and hashes,
excludes tracks whose audio ends before the Expert chart, and builds an
ephemeral worker-local audio manifest at training time. No local audio path or
copy belongs in a task view, experiment, bundle, or OCTAVE renderer payload.
Its training bundle has no inference profile. Promotion first recomputes its
declared song-disjoint validation split into a hash-bound held-out report, then
copies weights, configuration, and report into a separate immutable bundle.
There is intentionally no hard-coded score threshold: OCTAVE can show the
evidence and request the explicit promotion, but cannot treat a one-epoch
candidate as deployable.

The descriptor map is also STRUM's bundle-admission authority, not advisory
host metadata. The worker resolves the selected candidate before dispatch, and
each candidate writer revalidates the same contract after emission. The raw bundle must contain exactly its one mapped
component and no profile/companions; its manifest preprocessing and
hash-verified config must match the configured format, task kind, pipeline,
implementation, input/output semantics, and (for known-event candidates)
target semantics. A hash-valid combined or rehashed relabelled bundle is
rejected before the worker reports completion. This remains a raw experiment
gate and does not add Pro chart execution.

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
Descriptors whose existing held-out evaluation/package path is worker-ready
also publish `promotion_jobs`. Each job has a renderer-safe `options_schema`,
declared main-process-only request fields, output kind, and deployment scope;
OCTAVE invokes the selected one through `promotion start --json-events` rather
than hard-coding a profile command or exposing a path. The job adapter calls
the same strict evaluator/packager and cannot bypass its evidence or immutable
profile gates.
For `chart_transform.five_lane/v1`, each discoverable promotion job also
includes STRUM's immutable versioned quality policy. Its V1 song-disjoint
holdout minima are lane F1 0.50, precision 0.45, and recall 0.45: enough to
exclude the 0.3297 F1 baseline while avoiding precision-only or recall-only
promotion. Evaluation records the policy's canonical hash and decision;
packaging and profile validation recompute the same policy and reject absent,
failed, or altered evidence. It is intentionally not an OCTAVE option.
OCTAVE owns cancellation by terminating the supervised process group, rather
than requiring STRUM to retain a private-path job. A model folder is validated
separately from a STRUM runtime with `checkpoint discover`, `checkpoint
inspect`, `inference profile validate`, `chart preflight`, and `chart run`.
Discovery is a bounded private-folder scan that returns opaque bundle artifact
IDs and path-free, hash-verified profile metadata. It marks only profiles with
both a verified typed configuration and a declared chart handler as executable;
raw experiments and future capability contracts remain visible but
non-deployable. OCTAVE must retain its artifact-ID-to-folder lookup in the main
process and never pass a folder to the renderer.

Model-bundle profiles can additionally carry a `strum-profile-composition/v1`
graph. This is the safe declaration for a composed auto-chart profile: stages
name component IDs, companion runtime IDs/versions, dependency edges, typed
artifact inputs/outputs, and terminal per-instrument chart outputs. Bundle
validation proves the graph is acyclic and complete without resolving a source
or checkpoint path into a renderer payload. Preflight carries the resolved
graph so OCTAVE can display required companions and every stage. A graph is
not an execution adapter: unless STRUM implements the exact capability,
preflight leaves required stages `unavailable` and `chart run` rejects it.

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
`deployment_status: requires_keys_profile_evaluation_and_packaging` until
`strum-worker keys profile evaluate` has revalidated its held-out `PART KEYS`
view and `strum-worker keys profile package` has copied it into an immutable
`keys.neural-v1-expert/v1` bundle. The typed runtime consumes only
`keys.onset`/`keys.fret`, writes one Expert `PART KEYS` track, and cannot
select or substitute Guitar/Bass profiles.

The broader `strum.instrument-chart/bass/v1` and
`strum.instrument-chart/keys/v1` descriptors remain non-executable source
contracts for future architectures.  Their worker descriptors publish the
exact five-lane source track and the generic model, held-out evaluation,
profile-package, and chart-execution gates.  They also name the available,
separate V1 implementation path (`bass.onset-fret/v1` or
`keys.onset-fret/v1`) without aliasing task views, component identities, or
profiles.  OCTAVE may offer that concrete path for training, but cannot use a
generic descriptor as a fallback chart runtime.

### Vocal lead-component experiment gates

`vocals.note-activity/v1` is a catalog worker experiment, not a conversion of
vocal charts into five-lane data. Its `vocals_activity` task view is
revalidated against the selected catalog and may derive labels only from the
exact `PART VOCALS` track. The preprocessor produces 22.05 kHz / 128-mel
windows with two frame targets: pitched lead-vocal activity and MIDI pitch
36--84.

Before any lead-Vocal task view is written, STRUM applies its shared
`mido-standard-midi-exact-part-vocals/v1` predicate to every selected managed
MIDI asset. It excludes a record if mido cannot decode it or its decoded MIDI
does not contain exactly one `PART VOCALS` track. This is stricter than
import-time catalog coverage, is repeated during task-view resolution, and is
shared by the activity, phrase, lyric, and talky task kinds. The task view
records only the aggregate number excluded—never paths or source IDs—so the
four components cannot acquire incompatible target partitions silently.

Catalog inspection applies that same MIDI predicate and the shared
`soundfile-full-stream-decode/v1` preferred/fallback audio selection before
reporting eligibility. It returns aggregate target-versus-audio exclusion
counts only; a host can therefore make the preparation count actionable
without receiving source identifiers or local asset locations.

`strum-owned-lead-catalog-task-admission-resolver/v1` is a separate,
pre-model admission boundary. It receives a private catalog root and all four
lead task views, revalidates each immutable view against catalog assets, then
requires one identical source-id train/validation/test partition and exact
`PART VOCALS` selection across activity/pitch, phrase, lyric, and talky
components. STRUM parses those managed MIDI assets itself and publishes only
path-free source-set hashes, counts, and label-coverage outcomes. A missing
label category or insufficient/no-test coverage is `not_admitted`; a tampered
view or differing source partition is invalid. The resolver never loads a
checkpoint and is not a profile evaluator, packager, or chart handler.

The resulting `vocals.frame_activity_pitch` bundle has no profile and reports
`deployment_status: requires_vocals_profile_evaluation_and_packaging`. It
cannot be selected by `chart run`. A separate
`vocals.phrase-boundaries/v1` task derives start/end targets from lead-track
MIDI 105/106 markers or a sustained 105 phrase span, then writes only the
`vocals.phrase_boundaries` component. Its deployment status is
`requires_vocal_chart_composition_evaluation_and_packaging`, not a chart
handler. `vocals.lyric-alignment/v1` is a third bounded component: it trains a
character CTC acoustic encoder solely from observed `lyrics`/`text` meta
events on exact `PART VOCALS`, keeping the MIDI event timestamp alongside the
local target window. It neither imports external lyrics nor decides talkies,
harmonies, phrases, or chart structure. `vocals.talky-activity/v1` is a fourth
bounded component. Its labels are only duration-bearing note-96 spans on exact
`PART VOCALS`; note 96 is a pitchless/talky marker, not a sung MIDI pitch.
The trainer requires an observed note-96 span in both train and validation,
which prevents an all-negative evaluation from becoming a plausible model.
`HARM1`, `HARM2`, and `HARM3` are distinct source tracks and are deliberately
not collapsed into a lead target or trained from a shared vocal stem. The
catalog-preparation-only `vocals.harmony-source-policy/v1` worker path accepts
only OCTAVE's hash-bound `vocal-harmony-sources.json` sidecar. Each selected
track requires its own `harm1`, `harm2`, or `harm3` managed asset, the exact
decoded HARM MIDI track, and either an attested original isolated stem or a
separation output whose mix input, separator model, and configuration hashes
are all pinned. There is no `vocals`/`mix` fallback; a catalog without the
sidecar is ineligible. This produces a path-free source task only, not a
Harmony checkpoint, profile, or chart handler. The planned
`strum.instrument-chart/vocals/v1` descriptor preserves the remaining harmony
model, composition, held-out evaluation, packaging, and execution requirements
in a machine-readable non-executable contract. No Vocal worker path may invent
those outputs from a raw component.

The descriptor's `strum-vocal-chart-composition-contract/v1` additionally
defines the only valid future assembly boundary. All lead components must have
the same catalog-control lineage and source-ID partition and consume the same
catalog audio identity or a pinned same-master-timeline alignment. `PART
VOCALS` is assembled from separate pitch/activity, phrase, timestamped
lyric/text, and note-96 talky components; HARM tracks remain distinct and need
a future `vocals.harmony_model` trained only through the isolated-source
policy. A `strum-vocal-held-out-chart-evaluation-contract/v1` requires
source-disjoint test songs and STRUM-recomputed note, phrase, lyric/alignment,
talky, per-HARM-track, and assembled-MIDI evidence before a
`strum-vocal-profile-package-contract/v1` can bind components into a
`strum-profile-composition/v1` graph. The planned composition supports a
nonempty approved subset of `HARM1`/`HARM2`/`HARM3`, not an all-three
requirement, but records one exact binding per selected output: matching
`harmN` role, `strum-vocal-harmony-source-task/v1` task-view hash, OCTAVE
sidecar-policy hash, and catalog-control hash. The evaluator must repeat those
bindings alongside each HARM metric rather than reporting one aggregate
Harmony score. `strum-vocal-profile-quality-policy/v1` is STRUM-owned and
hash-pinned: it defines every metric and its all-required/all-selected-track
aggregation. Future packaging calls the contract validator to recompute
outcomes and rejects absent, unpinned, or failed policy evidence. No trainer,
package writer, or executable Vocal profile is introduced by that validation
contract. Until the named
`vocal_chart_profile_handler/v1` exists, execution is unavailable and fallback
to the legacy charter, shared Harmony audio, external lyrics, or raw component
outputs is forbidden.

## 11. Hardware

Developed on NVIDIA DGX Spark (GB10 GPU, CUDA 12.8). Inference runs in
~real-time on a single 12 GB GPU; training one drum classifier takes
~6–8 hours on the same hardware.
