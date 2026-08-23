# STRUM Roadmap

## Current contract status

STRUM is moving from legacy research scripts to a versioned, catalog-aware
worker runtime for OCTAVE. “Catalog-ready” means STRUM can create a safe,
path-free task view from OCTAVE's already-authorized catalog. It does **not**
mean the pipeline has a worker trainer, a deployable checkpoint, or a complete
auto-chart handler.

| Family | Catalog task view | Worker training | Executable worker profile |
| --- | --- | --- | --- |
| Guitar onset + fret | Available | Available; experiment-only | `guitar.hybrid-v2-rule/v1`, or evaluated `guitar.neural-v1-expert/v1`; Expert only |
| Bass onset + fret | Available | Available; raw experiment gated | Evaluated `bass.neural-v1-expert/v1`; Expert only |
| Keys onset + fret | Available | Available; raw experiment gated | Evaluated `keys.neural-v1-expert/v1`; Expert only |
| Vocal lead components (activity/pitch, phrase, lyric, talky) | Available | Available; experiment-only | Planned; requires harmony, composition, and Vocal evaluation |
| Drums onset + classifier | Available | Available; V2 evaluation package | `drums.v14-expert/v1` direct Expert charts; V2 is evaluation-only |
| Five-lane difficulty transform (Guitar/Bass/Keys/Drums) | Available | Available | `difficulty.transform/v1` |
| Generic Keys / Vocals / Pro Guitar / Pro Bass / Pro Keys | Available | Planned | Planned |
| Guitar/Bass fret mapper | Available | Available; requires `pitch` extra and records exact Basic Pitch provenance | Planned; requires evaluation/profile packaging |
| Guitar/Bass section classifier | Available | Available; experiment-only | Planned; requires section-routing evaluation and runtime profile |

The legacy all-instrument batch scripts remain research/compatibility tools;
they are not a single deployable worker profile. Their behavior must not be
presented as a successful OCTAVE auto-chart result.

## Shipped

- ✅ Versioned worker discovery: runtime probe, dynamic pipeline descriptors,
  catalog inspection, safe task-view preparation, bounded path-free
  checkpoint-folder discovery/inspection, profile validation, chart preflight,
  and chart execution. A discovered bundle is deployable only when a typed
  profile validates its companion hashes and configuration and STRUM declares
  a matching chart handler.
- ✅ OCTAVE catalog boundary: STRUM consumes only
  `octave-song-source-catalog/v1` managed assets marked
  `training_use: allowed`; it never reads imported package/source locations.
- ✅ Path-free catalog task views and lineage for Guitar, Drums, Bass, Keys,
  Vocals, Pro instruments, fret mapper, and section families, including
  declared event/label-source schemas and catalog-validated MIDI track
  selection for every future trainer family.
- ✅ Pro task views now use exact REAL_* track identities rather than prefix
  matching. Expert Pro Keys uses only `PART REAL_KEYS_X`; Pro Guitar/Bass
  preserve standard and `_22` source variants. Prepare now produces strict,
  path-free decoded Pro targets (string/fret/technique or pitch/range shifts)
  plus a catalog-revalidated, global-tempo-correct audio event-window cache.
  The cache is research-only; descriptors remain non-executable until STRUM
  supplies an event trainer, held-out evaluator, package, and chart handler.
- ✅ Worker lifecycle streams for dataset preparation and training. OCTAVE
  owns process creation/cancellation and retains private paths.
- ✅ Bundle-validated Expert Guitar hybrid profile and Expert Drums direct V14
  profile. Both fail closed rather than invoking legacy companion/fallback
  behavior.
- ✅ Typed, path-free chart preflight and result manifests for the executable
  Guitar, Drums, and learned difficulty-transform profiles. Per-instrument
  component stages state `ready`, `provided`, `succeeded`, `not_requested`,
  or `unavailable`; the selected difficulty policy is always explicit.
- ✅ Versioned, path-free composed-profile graph declarations. A bundle can
  expose every required stage, component, versioned runtime companion,
  dependency edge, and terminal chart output through `checkpoint inspect` and
  chart preflight. Graphs with no registered STRUM handler remain explicitly
  non-executable; they cannot fall back to the legacy batch assembly path.
- ✅ Catalog-backed, worker-trainable learned five-lane chart transforms for
  Guitar, Bass, Keys, and Drums, including explicit Expert → lower-difficulty
  provenance. The optional `rms_onset_v1` song-conditioning baseline is
  selected during task preparation, has catalog audio hashes/roles and
  duration alignment verified before training, and uses only temporary
  worker-local audio copies.
- ✅ Catalog-backed, worker-trainable Guitar/Bass/Keys onset/fret and Drums
  onset classifier experiments. Guitar, Bass, and Keys reuse the same
  five-lane feature extractor only through their instrument-specific,
  revalidated `PART GUITAR`, `PART BASS`, and `PART KEYS` task views; their
  components and experiment configuration IDs remain distinct. All raw
  experiments are non-deployable. Guitar V1 has an explicit
  validation/packaging path. Bass now has its own held-out `PART BASS`
  evaluator and hash-verified Expert-only profile/runtime; Keys has the same
  distinct held-out evaluator and Expert-only runtime boundary.
- ✅ Catalog-backed, worker-trainable Vocal frame activity/sung-pitch,
  lead-phrase-boundary, observed-lyric/alignment, and pitchless/talky
  experiments. The exact
  `PART VOCALS` source tasks emit separate `vocals.frame_activity_pitch`,
  `vocals.phrase_boundaries`, `vocals.lyric_alignment`, and
  `vocals.talky_activity` components; no raw component is a profile. Talky
  labels are only source note-96 spans and require positives in train and val.
  The current 58-song OCTAVE curated catalog contains only two lead-talky
  sources. A declared `sha256-source-id-seed-mod-100/v2` 67/33 smoke split can
  place one positive in each split and exercise the worker, but its zero-F1
  one-epoch result is not quality evidence; curation still needs substantially
  more held-out talky examples.
  The planned generic Vocal descriptor retains harmony, composition, held-out
  evaluation, packaging, and execution stages.
- ✅ Catalog-backed, worker-trainable Guitar/Bass fret-mapper experiments.
  They require the `pitch` extra, preserve catalog train/validation splits,
  record the Basic Pitch distribution/version that generated their features,
  and remain profile-gated components rather than legacy fallback behavior.
  Their immutable release requirements explicitly block promotion until STRUM
  has an exact onset composition, tensor-only strict loader, pinned Viterbi
  policy, and end-to-end held-out chart evaluation.
- ✅ Catalog-backed, worker-trainable Guitar/Bass section-classifier
  experiments. STRUM derives the six chart-pattern labels from the declared
  `PART GUITAR` or `PART BASS` task-view track, revalidates the catalog split
  while materializing log-mel caches, and packages a hash-verified
  `section_classifier.{instrument}` component with no inference profile. The
  artifact records the exact legacy `SectionRouter` librosa frontend contract
  rather than relying on matching mel dimensions and labels. A strict
  tensor-only candidate loader and validation-calibrated, test-only
  `section.classifier-evaluation/v1` profile are now available for evidence
  collection only. Promotion still requires a router-on/off chart-impact
  ablation, composition with a concrete Guitar or Bass auto-chart profile,
  and a registered chart-execution handler.
- ✅ A catalog-worker Drums V2 experiment can be integrity-packaged as
  `drums.onset-classifier-evaluation/v1`. Its hash-verified runtime accepts
  only STRUM's prepared onset windows and returns eight class probabilities;
  it cannot detect onsets, emit velocity, write a chart, or replace the direct
  `drums.v14-expert/v1` profile.
- ✅ Legacy model/training research assets: two-stage drums, Guitar onset,
  mapper, section, vocals, keys, tempo/grid, MIDI export, and batch assembly.

## Next worker milestones

1. Calibrate/evaluate Guitar, Bass, and Keys V1 on useful held-out catalogs and
   establish complete instrument-specific profiles. Guitar's packaging path
   verifies architecture, preprocessing, component hashes, tensor-only state
   dictionaries, and a revalidated validation report. Bass now has equivalent
   instrument-specific evaluation and packaging without reusing a Guitar
   profile; Keys now has an equivalent Keys-only boundary. The current V2 Drums package is intentionally Stage-2
   evaluation only; no artifact is promoted solely because its checkpoint
   exists.
2. Replace the remaining legacy auto-chart assembly behavior with declared,
   component-level bundle requirements and typed stage graphs. The current
   typed manifests cover each executable single-profile run; the full
   multi-instrument graph still needs the same partial-run/fallback reporting.
3. Add dedicated learned trainers and event schemas for the generic Keys task,
   Pro instruments, and section routing. Evaluate the bounded section
   classifier against section-routing utility before assigning it a runtime
   profile. Promote the bounded Vocal and fret-mapper experiments only after
   their remaining component stages, held-out evaluation, and complete profile
   contracts exist.
4. Improve the learned difficulty model from the current event baseline to
   audio/stem-aware sequence modeling with song-disjoint evaluation. Difficulty
   mapping stays in STRUM; OCTAVE only selects and displays a validated profile.
5. Publish signed/versioned runtime and model-bundle releases, then support
   OCTAVE-managed immutable runtime installation, locking, rollback, and
   offline use. Release OCTAVE must not auto-pull arbitrary Git branches.
6. Add shared worker fixtures and real-catalog end-to-end smoke tests proving
   source paths never appear in a task view, experiment, checkpoint, renderer
   payload, event stream, log, or user-visible error.

## Research and evaluation

- Per-profile, song-disjoint evaluation with declared operating envelopes and
  bundle provenance.
- Genre-specific Drums variants and community-authorized training data.
- Streaming inference, Pro Guitar string/fret inference, and a web demo only
  after their model/profile contracts are explicit.

## Hardware

- **Training**: CUDA-capable NVIDIA GPU preferred; the runtime also declares
  MPS and CPU support where a profile permits it.
- **Inference**: profile requirements, optional dependencies, and device
  support are discovered through `strum-worker probe --json` and each bundle
  preflight rather than assumed from a legacy script.
