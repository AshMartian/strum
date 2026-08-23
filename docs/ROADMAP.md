# STRUM Roadmap

## Current contract status

STRUM is moving from legacy research scripts to a versioned, catalog-aware
worker runtime for OCTAVE. “Catalog-ready” means STRUM can create a safe,
path-free task view from OCTAVE's already-authorized catalog. It does **not**
mean the pipeline has a worker trainer, a deployable checkpoint, or a complete
auto-chart handler.

| Family | Catalog task view | Worker training | Executable worker profile |
| --- | --- | --- | --- |
| Guitar onset + fret | Available | Script-only | `guitar.hybrid-v2-rule/v1`, Expert only |
| Drums onset + classifier | Available | Script-only | `drums.v14-expert/v1`, Expert only direct V14 |
| Five-lane difficulty transform (Guitar/Bass/Keys/Drums) | Available | Available | `difficulty.transform/v1` |
| Bass / Keys / Vocals / Pro Guitar / Pro Bass / Pro Keys | Available | Planned | Planned |
| Guitar/Bass fret mapper and section classifier | Available | Planned worker handler | Planned |

The legacy all-instrument batch scripts remain research/compatibility tools;
they are not a single deployable worker profile. Their behavior must not be
presented as a successful OCTAVE auto-chart result.

## Shipped

- ✅ Versioned worker discovery: runtime probe, dynamic pipeline descriptors,
  catalog inspection, safe task-view preparation, model/checkpoint inspection,
  profile validation, chart preflight, and chart execution.
- ✅ OCTAVE catalog boundary: STRUM consumes only
  `octave-song-source-catalog/v1` managed assets marked
  `training_use: allowed`; it never reads imported package/source locations.
- ✅ Path-free catalog task views and lineage for Guitar, Drums, Bass, Keys,
  Vocals, Pro instruments, fret mapper, and section families.
- ✅ Worker lifecycle streams for dataset preparation and training. OCTAVE
  owns process creation/cancellation and retains private paths.
- ✅ Bundle-validated Expert Guitar hybrid profile and Expert Drums direct V14
  profile. Both fail closed rather than invoking legacy companion/fallback
  behavior.
- ✅ Catalog-backed, worker-trainable learned five-lane chart transforms for
  Guitar, Bass, Keys, and Drums, including explicit Expert → lower-difficulty
  provenance.
- ✅ Legacy model/training research assets: two-stage drums, Guitar onset,
  mapper, section, vocals, keys, tempo/grid, MIDI export, and batch assembly.

## Next worker milestones

1. Make Guitar and Drums catalog trainers worker-runable: typed training
   schemas, cache/preprocessing jobs, experiment manifests, compatible
   resume/fine-tune checks, evaluation output, and deployable bundle profiles.
2. Replace the remaining legacy auto-chart assembly behavior with declared,
   component-level bundle requirements and typed result manifests. A partial
   run must state its actual stage status and fallback policy.
3. Add dedicated learned trainers and event schemas for Bass, Keys, Vocals,
   Pro instruments, fret mapping, and section routing. Their existing catalog
   views are the input boundary, not implementation completion.
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
