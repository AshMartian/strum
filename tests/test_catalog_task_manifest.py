from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

from src.catalog_task_manifest import (
    LEGACY_SPLIT_ALGORITHM,
    MANIFEST_FORMAT,
    PIPELINE_IDS,
    available_task_kinds,
    build_catalog_task_manifest,
    deterministic_split,
    resolve_catalog_task_manifest_songs,
)
from src.five_lane_runtime_admission import (
    PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT,
    classify_five_lane_runtime_source,
)
from src.song_source_catalog import CatalogValidationError
from src.vocal_audio_compatibility import has_compatible_vocal_audio


def _asset(root: Path, payload: bytes, filename: str) -> dict[str, object]:
    digest = hashlib.sha256(payload).hexdigest()
    relative = f"assets/sha256/{digest}/{filename}"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "asset_id": f"sha256:{digest}",
        "sha256": digest,
        "relative_path": relative,
        "byte_length": len(payload),
        "media_type": None,
    }


def _five_lane_midi() -> bytes:
    midi = mido.MidiFile()
    for track_name in ("PART GUITAR", "PART BASS"):
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=track_name, time=0))
        track.append(mido.Message("note_on", note=96, velocity=100, time=0))
        track.append(mido.Message("note_off", note=96, velocity=0, time=480))
    vocals = mido.MidiTrack()
    midi.tracks.append(vocals)
    vocals.append(mido.MetaMessage("track_name", name="PART VOCALS", time=0))
    vocals.append(mido.Message("note_on", note=60, velocity=100, time=0))
    vocals.append(mido.Message("note_off", note=60, velocity=0, time=480))
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue()


def _midi_with_out_of_range_data_byte() -> bytes:
    """Return a chunk-layout-valid MIDI whose note velocity violates SMF."""
    return (
        b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60"
        b"MTrk\x00\x00\x00\x08\x00\x90\x3c\xc8\x00\xff\x2f\x00"
    )


def _wav_bytes() -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.full(512, 0.01, dtype=np.float32), 22_050, format="WAV")
    return stream.getvalue()


def _record(root: Path, source_id: str, *, training_use: str = "allowed") -> dict[str, object]:
    track_names = {
        "bass": ["PART BASS"],
        "keys": ["PART KEYS"],
        "vocals": ["PART VOCALS"],
        "pro_guitar": ["PART REAL_GUITAR"],
        "pro_bass": ["PART REAL_BASS_22"],
        "pro_keys": ["PART REAL_KEYS_X"],
        "guitar": ["PART GUITAR"],
    }
    instruments = {
        instrument: {
            "status": "present",
            "difficulties": ["expert"],
            "track_names": names,
        }
        for instrument, names in track_names.items()
    }
    audio_payloads = {
        # The five generic task kinds only need content-addressed bytes here,
        # but every Vocal task now tests the actual shared decoder boundary.
        "mix": _wav_bytes(),
        "vocals": _wav_bytes(),
    }
    return {
        "source_id": source_id,
        "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
        "rights": {
            "training_use": training_use,
            "provenance": "Reviewed local collection",
            "license": "test-only",
        },
        "metadata": {"name": "Safe Song"},
        "chart": {
            "notes_midi": _asset(root, _five_lane_midi(), "notes.mid"),
            "instruments": instruments,
        },
        "audio": {
            role: _asset(
                root,
                audio_payloads.get(role, f"{source_id}-{role}".encode()),
                f"{role}.ogg",
            )
            for role in ("mix", "bass", "guitar", "keys", "vocals")
        },
    }


def _catalog(root: Path, records: list[dict[str, object]]) -> None:
    (root / "records.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "local-test",
                "records": "records.jsonl",
            }
        )
    )


def _bass_midi_without_expert_notes() -> bytes:
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="PART BASS", time=0))
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue()


def test_runtime_admission_keeps_prepare_preprocess_and_evaluation_inputs_in_parity(
    tmp_path: Path,
) -> None:
    good = _record(tmp_path, "octave-src-aaaaaaaa")
    good["audio"]["bass"] = _asset(tmp_path, _wav_bytes(), "bass.wav")
    unreadable = _record(tmp_path, "octave-src-bbbbbbbb")
    unreadable["audio"]["bass"] = _asset(tmp_path, b"not-audio", "bass.ogg")
    missing_expert = _record(tmp_path, "octave-src-cccccccc")
    missing_expert["audio"]["bass"] = _asset(tmp_path, _wav_bytes(), "bass.wav")
    missing_expert["chart"]["notes_midi"] = _asset(
        tmp_path, _bass_midi_without_expert_notes(), "notes.mid"
    )
    _catalog(tmp_path, [good, unreadable, missing_expert])

    manifest = build_catalog_task_manifest(tmp_path, "bass_onset_fret", runtime_admission=True)

    assert [song["source_id"] for song in manifest["songs"]] == ["octave-src-aaaaaaaa"]
    assert manifest["summary"]["runtime_admission"] == {
        "format": "strum-five-lane-runtime-admission/v1",
        "exclusion_reason_counts": {
            "runtime_audio_unreadable": 1,
            "exact_expert_label_missing": 1,
        },
    }
    resolved = resolve_catalog_task_manifest_songs(manifest, tmp_path)
    assert len(resolved) == 1
    source = resolved[0]
    assert (
        classify_five_lane_runtime_source(
            Path(str(source["audio_path"])),
            Path(str(source["midi_path"])),
            label_track="PART BASS",
        )
        is None
    )


def test_runtime_admission_is_rejected_for_non_profile_task_views(tmp_path: Path) -> None:
    with pytest.raises(CatalogValidationError, match="only supported"):
        build_catalog_task_manifest(tmp_path, "vocals_activity", runtime_admission=True)

    record = _record(tmp_path, "octave-src-aaaaaaaa")
    _catalog(tmp_path, [record])
    manifest = build_catalog_task_manifest(tmp_path, "vocals_activity")
    manifest["task"]["runtime_admission"] = "strum-five-lane-runtime-admission/v1"
    with pytest.raises(CatalogValidationError, match="runtime admission"):
        resolve_catalog_task_manifest_songs(manifest, tmp_path)


@pytest.mark.parametrize(
    ("task_kind", "instrument"),
    (("bass_onset_fret", "bass"), ("keys_onset_fret", "keys")),
)
def test_profile_grade_task_views_bind_dedicated_audio_and_reject_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_kind: str,
    instrument: str,
) -> None:
    monkeypatch.setitem(PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT, "train", 1)
    monkeypatch.setitem(PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT, "val", 1)
    monkeypatch.setitem(PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT, "test", 1)
    monkeypatch.setattr(
        "src.catalog_task_manifest.classify_five_lane_runtime_source",
        lambda *_args, **_kwargs: None,
    )
    found: dict[str, str] = {}
    for index in range(10_000):
        source_id = f"octave-src-profile-{instrument}-{index:08d}"
        found.setdefault(deterministic_split(source_id, seed="catalog-source-id/v1"), source_id)
        if len(found) == 3:
            break
    _catalog(tmp_path, [_record(tmp_path, found[split]) for split in ("train", "val", "test")])

    manifest = build_catalog_task_manifest(
        tmp_path,
        task_kind,
        disable_fallback=True,
        profile_grade=True,
    )

    assert manifest["task"]["audio_role"] == instrument
    assert manifest["task"]["fallback_audio_role"] is None
    assert all(song["audio_role"] == instrument for song in manifest["songs"])
    assert manifest["summary"]["profile_grade_admission"]["meets_minimums"] is True
    assert str(tmp_path) not in json.dumps(manifest)

    manifest["task"]["fallback_audio_role"] = "mix"
    with pytest.raises(CatalogValidationError, match="profile-grade admission"):
        resolve_catalog_task_manifest_songs(manifest, tmp_path)


def test_profile_grade_is_unavailable_to_non_five_lane_catalog_tasks(tmp_path: Path) -> None:
    with pytest.raises(CatalogValidationError, match="only supported"):
        build_catalog_task_manifest(tmp_path, "vocals_activity", profile_grade=True)


def test_vocal_audio_gate_rejects_decode_failure_after_open_and_initial_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MiddleDecodeFailure:
        frames = 12_288
        samplerate = 22_050
        channels = 1
        reads = 0

        def __enter__(self) -> MiddleDecodeFailure:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self, frames: int, **_: object) -> np.ndarray:
            self.reads += 1
            if self.reads == 1:
                return np.ones(frames, dtype=np.float32)
            raise RuntimeError("synthetic middle-stream decoder failure")

    monkeypatch.setattr(
        "src.vocal_audio_compatibility.sf.SoundFile",
        lambda *_args, **_kwargs: MiddleDecodeFailure(),
    )

    assert has_compatible_vocal_audio(tmp_path / "hash-valid-but-truncated.ogg") is False


@pytest.mark.parametrize("task_kind", available_task_kinds())
def test_all_remaining_training_families_use_one_path_free_catalog_contract(
    tmp_path: Path, task_kind: str
) -> None:
    _catalog(
        tmp_path,
        [
            _record(tmp_path, "octave-src-aaaaaaaa"),
            _record(tmp_path, "octave-src-bbbbbbbb", training_use="review_required"),
        ],
    )

    manifest = build_catalog_task_manifest(
        tmp_path,
        task_kind,
        preprocessing={"sample_rate": 22050, "window_seconds": 2},
    )

    serialized = json.dumps(manifest)
    assert manifest["format"] == MANIFEST_FORMAT
    assert manifest["task"]["pipeline_id"] == PIPELINE_IDS[task_kind]
    assert manifest["task"]["label_schema"]["id"]
    assert manifest["lineage"]["catalog_control_sha256"]
    assert manifest["task"]["preprocessing_sha256"]
    assert [song["source_id"] for song in manifest["songs"]] == ["octave-src-aaaaaaaa"]
    assert str(tmp_path) not in serialized
    assert "provenance" not in serialized

    resolved = resolve_catalog_task_manifest_songs(manifest, tmp_path)
    assert resolved[0]["source_id"] == "octave-src-aaaaaaaa"
    assert resolved[0]["label_schema"] == manifest["task"]["label_schema"]
    assert resolved[0]["label_tracks"] == manifest["songs"][0]["label_tracks"]
    assert str(tmp_path) in resolved[0]["audio_path"]


def test_rejects_task_label_schema_or_track_tampering(tmp_path: Path) -> None:
    _catalog(tmp_path, [_record(tmp_path, "octave-src-aaaaaaaa")])
    manifest = build_catalog_task_manifest(tmp_path, "pro_guitar")

    manifest["task"]["label_schema"]["id"] = "five-lane-midi/v1"
    with pytest.raises(CatalogValidationError, match="label schema"):
        resolve_catalog_task_manifest_songs(manifest, tmp_path)

    manifest = build_catalog_task_manifest(tmp_path, "pro_guitar")
    manifest["songs"][0]["label_tracks"] = ["PART GUITAR"]
    with pytest.raises(CatalogValidationError, match="valid approved catalog task input"):
        resolve_catalog_task_manifest_songs(manifest, tmp_path)


def test_split_seed_changes_new_task_assignment_and_legacy_views_still_resolve(
    tmp_path: Path,
) -> None:
    source_ids = [f"octave-src-{index:08x}" for index in range(1000)]
    chosen = next(
        source_id
        for source_id in source_ids
        if deterministic_split(source_id, seed="seed-a")
        != deterministic_split(source_id, seed="seed-b")
    )
    _catalog(tmp_path, [_record(tmp_path, chosen)])

    first = build_catalog_task_manifest(tmp_path, "section_guitar", split_seed="seed-a")
    second = build_catalog_task_manifest(tmp_path, "section_guitar", split_seed="seed-b")

    assert first["task"]["split_algorithm"] != LEGACY_SPLIT_ALGORITHM
    assert first["songs"][0]["split"] != second["songs"][0]["split"]
    assert (
        resolve_catalog_task_manifest_songs(first, tmp_path)[0]["split"]
        == first["songs"][0]["split"]
    )

    legacy = build_catalog_task_manifest(tmp_path, "section_guitar", split_seed="ignored-by-v1")
    legacy["task"]["split_algorithm"] = LEGACY_SPLIT_ALGORITHM
    legacy["songs"][0]["split"] = deterministic_split(chosen)

    assert (
        resolve_catalog_task_manifest_songs(legacy, tmp_path)[0]["split"]
        == legacy["songs"][0]["split"]
    )


@pytest.mark.parametrize("task_kind", ("section_guitar", "section_bass"))
def test_section_task_views_exclude_malformed_midi_sources(tmp_path: Path, task_kind: str) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    record["chart"]["notes_midi"] = _asset(tmp_path, b"not a midi file", "notes.mid")
    _catalog(tmp_path, [record])

    manifest = build_catalog_task_manifest(tmp_path, task_kind)

    assert manifest["songs"] == []
    assert manifest["summary"]["record_count"] == 0


@pytest.mark.parametrize(
    ("task_kind", "instrument", "case_variant"),
    [
        ("section_guitar", "guitar", "part guitar"),
        ("section_bass", "bass", "Part Bass"),
    ],
)
def test_section_prepare_requires_the_exact_canonical_label_track(
    tmp_path: Path,
    task_kind: str,
    instrument: str,
    case_variant: str,
) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    record["chart"]["instruments"][instrument]["track_names"] = [case_variant]
    _catalog(tmp_path, [record])

    with pytest.raises(CatalogValidationError, match="no track"):
        build_catalog_task_manifest(tmp_path, task_kind)


def test_section_prefix_schema_views_must_be_reprepared(tmp_path: Path) -> None:
    _catalog(tmp_path, [_record(tmp_path, "octave-src-aaaaaaaa")])
    manifest = build_catalog_task_manifest(tmp_path, "section_guitar")
    manifest["task"]["label_schema"] = {
        "id": "midi-section-events/v1",
        "track_prefixes": ["PART GUITAR"],
        "difficulty_encoding": "not-applicable",
    }

    with pytest.raises(CatalogValidationError, match="retired prefix.*re-prepare"):
        resolve_catalog_task_manifest_songs(manifest, tmp_path)


@pytest.mark.parametrize(
    "task_kind",
    (
        "vocals",
        "vocals_activity",
        "vocals_phrase_boundaries",
        "vocals_lyric_alignment",
        "vocals_talky_activity",
    ),
)
def test_vocal_task_views_exclude_midi_targets_mido_cannot_decode(
    tmp_path: Path, task_kind: str
) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    # OCTAVE's import-time chunk check can accept this payload, but its velocity
    # byte is outside the Standard MIDI File 0..127 data-byte range. A correct
    # STRUM task view must not defer that failure to one component preprocessor.
    malformed = _midi_with_out_of_range_data_byte()
    with pytest.raises(OSError, match="data byte"):
        mido.MidiFile(file=io.BytesIO(malformed))
    record["chart"]["notes_midi"] = _asset(tmp_path, malformed, "notes.mid")
    _catalog(tmp_path, [record])

    manifest = build_catalog_task_manifest(tmp_path, task_kind)

    assert manifest["songs"] == []
    assert manifest["summary"] == {
        "record_count": 0,
        "by_split": {},
        "target_compatibility": {
            "format": "mido-standard-midi-exact-part-vocals/v1",
            "excluded_record_count": 1,
            "audio": {
                "format": "soundfile-full-stream-decode/v1",
                "excluded_record_count": 0,
            },
        },
    }


def test_vocal_task_views_require_one_actual_exact_part_vocals_track(tmp_path: Path) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    # Import-time coverage cannot override STRUM's exact target semantics: a
    # chart with only guitar events is not a Vocal label source.
    midi = mido.MidiFile()
    guitar = mido.MidiTrack()
    midi.tracks.append(guitar)
    guitar.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
    guitar.append(mido.Message("note_on", note=96, velocity=100, time=0))
    guitar.append(mido.Message("note_off", note=96, velocity=0, time=1))
    output = io.BytesIO()
    midi.save(file=output)
    record["chart"]["notes_midi"] = _asset(tmp_path, output.getvalue(), "notes.mid")
    _catalog(tmp_path, [record])

    manifest = build_catalog_task_manifest(tmp_path, "vocals_activity")

    assert manifest["songs"] == []
    assert manifest["summary"]["target_compatibility"]["excluded_record_count"] == 1


@pytest.mark.parametrize(
    "task_kind",
    (
        "vocals",
        "vocals_activity",
        "vocals_phrase_boundaries",
        "vocals_lyric_alignment",
        "vocals_talky_activity",
    ),
)
def test_vocal_task_views_exclude_decoder_incompatible_audio_before_training(
    tmp_path: Path, task_kind: str
) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    # A hash-valid catalog asset is not necessarily a libsndfile-decodable
    # audio stream.  With fallback disabled every lead Vocal target kind must
    # leave it out rather than letting its later preprocessor abort the job.
    record["audio"]["vocals"] = _asset(tmp_path, b"not-decodable-audio", "vocals.ogg")
    _catalog(tmp_path, [record])

    manifest = build_catalog_task_manifest(tmp_path, task_kind, disable_fallback=True)

    assert manifest["songs"] == []
    assert manifest["summary"]["target_compatibility"] == {
        "format": "mido-standard-midi-exact-part-vocals/v1",
        "excluded_record_count": 0,
        "audio": {
            "format": "soundfile-full-stream-decode/v1",
            "excluded_record_count": 1,
        },
    }
    assert str(tmp_path) not in json.dumps(manifest)


def test_vocal_decoder_boundary_uses_declared_fallback_before_excluding_source(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    record["audio"]["vocals"] = _asset(tmp_path, b"not-decodable-audio", "vocals.ogg")
    _catalog(tmp_path, [record])

    manifest = build_catalog_task_manifest(tmp_path, "vocals_activity")

    assert manifest["songs"][0]["audio_role"] == "mix"
    assert manifest["summary"]["target_compatibility"]["audio"]["excluded_record_count"] == 0
    assert resolve_catalog_task_manifest_songs(manifest, tmp_path)[0]["audio_kind"] == "mix"


def test_vocal_runtime_rechecks_decoder_compatibility_for_forged_or_stale_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    record["audio"]["vocals"] = _asset(tmp_path, b"not-decodable-audio", "vocals.ogg")
    _catalog(tmp_path, [record])
    # Simulate a task view produced before this strict gate existed (or forged
    # by a caller).  Catalog identity remains valid, so re-resolution itself
    # must reject the audio instead of sending it to a late preprocessor.
    monkeypatch.setattr("src.catalog_task_manifest.has_compatible_vocal_audio", lambda _: True)
    manifest = build_catalog_task_manifest(tmp_path, "vocals_activity", disable_fallback=True)
    assert manifest["songs"]
    monkeypatch.undo()

    with pytest.raises(CatalogValidationError, match="valid approved catalog task input"):
        resolve_catalog_task_manifest_songs(manifest, tmp_path)


@pytest.mark.parametrize(
    "task_kind",
    (
        "vocals",
        "vocals_activity",
        "vocals_phrase_boundaries",
        "vocals_lyric_alignment",
        "vocals_talky_activity",
    ),
)
def test_vocal_task_views_ignore_prefix_matched_alt_coverage_tracks(
    tmp_path: Path, task_kind: str
) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    # The actual MIDI fixture has only canonical PART VOCALS. Import coverage
    # may mention an ALT arrangement too, but the task source is never allowed
    # to merge it through the legacy activity schema's prefix declaration.
    record["chart"]["instruments"]["vocals"]["track_names"] = [
        "PART VOCALS",
        "PART VOCALS ALT",
    ]
    _catalog(tmp_path, [record])

    manifest = build_catalog_task_manifest(tmp_path, task_kind)

    assert manifest["songs"][0]["label_tracks"] == ["PART VOCALS"]
    assert resolve_catalog_task_manifest_songs(manifest, tmp_path)[0]["label_tracks"] == [
        "PART VOCALS"
    ]


def test_vocal_runtime_rejects_stale_or_forged_incompatible_target(tmp_path: Path) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    record["chart"]["notes_midi"] = _asset(
        tmp_path, _midi_with_out_of_range_data_byte(), "notes.mid"
    )
    _catalog(tmp_path, [record])
    manifest = build_catalog_task_manifest(tmp_path, "vocals_activity")
    asset = record["chart"]["notes_midi"]
    assert isinstance(asset, dict)
    manifest["songs"] = [
        {
            "source_id": "octave-src-aaaaaaaa",
            "instrument": "vocals",
            "required_difficulty": "expert",
            "split": deterministic_split("octave-src-aaaaaaaa", seed="catalog-source-id/v1"),
            "audio_role": "vocals",
            "label_tracks": ["PART VOCALS"],
            "audio": record["audio"]["vocals"],
            "notes_midi": asset,
        }
    ]

    with pytest.raises(CatalogValidationError, match="valid approved catalog task input"):
        resolve_catalog_task_manifest_songs(manifest, tmp_path)


def test_pro_task_views_keep_exact_real_track_semantics(tmp_path: Path) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    # Prefix matching must not turn similarly named chart/authoring tracks
    # into Pro training labels, and an Expert Pro Keys task must not consume
    # lower-difficulty REAL_KEYS tracks.
    record["chart"]["instruments"]["pro_guitar"]["track_names"] = [
        "PART REAL_GUITAR",
        "PART REAL_GUITAR_22",
        "PART REAL_GUITAR_PREVIEW",
    ]
    record["chart"]["instruments"]["pro_bass"]["track_names"] = [
        "PART REAL_BASS_22",
        "PART REAL_BASS_AUTHORING",
    ]
    record["chart"]["instruments"]["pro_keys"]["track_names"] = [
        "PART REAL_KEYS_E",
        "PART REAL_KEYS_M",
        "PART REAL_KEYS_H",
        "PART REAL_KEYS_X",
        "PART REAL_KEYS_X_PREVIEW",
    ]
    _catalog(tmp_path, [record])

    guitar = build_catalog_task_manifest(tmp_path, "pro_guitar")
    assert guitar["songs"][0]["label_tracks"] == ["PART REAL_GUITAR", "PART REAL_GUITAR_22"]
    bass = build_catalog_task_manifest(tmp_path, "pro_bass")
    assert bass["songs"][0]["label_tracks"] == ["PART REAL_BASS_22"]
    keys = build_catalog_task_manifest(tmp_path, "pro_keys")
    assert keys["songs"][0]["label_tracks"] == ["PART REAL_KEYS_X"]
    assert keys["task"]["label_schema"]["track_names"] == ["PART REAL_KEYS_X"]

    keys["songs"][0]["label_tracks"] = ["PART REAL_KEYS_H"]
    with pytest.raises(CatalogValidationError, match="valid approved catalog task input"):
        resolve_catalog_task_manifest_songs(keys, tmp_path)


@pytest.mark.parametrize(
    ("task_kind", "instrument", "exact_track", "alternate_track"),
    [
        ("bass", "bass", "PART BASS", "PART BASS ALT"),
        ("bass_onset_fret", "bass", "PART BASS", "PART BASS ALT"),
        ("keys", "keys", "PART KEYS", "PART KEYS ALT"),
        ("keys_onset_fret", "keys", "PART KEYS", "PART KEYS ALT"),
    ],
)
def test_exact_five_lane_task_views_do_not_select_alternate_arrangements(
    tmp_path: Path,
    task_kind: str,
    instrument: str,
    exact_track: str,
    alternate_track: str,
) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    record["chart"]["instruments"][instrument]["track_names"] = [
        exact_track,
        alternate_track,
    ]
    _catalog(tmp_path, [record])

    manifest = build_catalog_task_manifest(tmp_path, task_kind)

    assert manifest["task"]["label_schema"] == {
        "id": "five-lane-midi/v2",
        "track_names": [exact_track],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    }
    assert manifest["songs"][0]["label_tracks"] == [exact_track]
    assert resolve_catalog_task_manifest_songs(manifest, tmp_path)[0]["label_tracks"] == [
        exact_track
    ]


@pytest.mark.parametrize(
    ("task_kind", "instrument", "exact_track", "alternate_track"),
    [
        ("bass_onset_fret", "bass", "PART BASS", "PART BASS ALT"),
        ("keys_onset_fret", "keys", "PART KEYS", "PART KEYS ALT"),
    ],
)
def test_legacy_five_lane_task_views_only_resolve_when_their_selection_is_exact(
    tmp_path: Path,
    task_kind: str,
    instrument: str,
    exact_track: str,
    alternate_track: str,
) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    record["chart"]["instruments"][instrument]["track_names"] = [
        exact_track,
        alternate_track,
    ]
    _catalog(tmp_path, [record])
    manifest = build_catalog_task_manifest(tmp_path, task_kind)
    manifest["task"]["label_schema"] = {
        "id": "five-lane-midi/v1",
        "track_prefixes": [exact_track],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    }

    # Existing V1 task views remain usable when they already selected the
    # single exact track consumed by the preprocessor.
    assert resolve_catalog_task_manifest_songs(manifest, tmp_path)[0]["label_tracks"] == [
        exact_track
    ]

    manifest["songs"][0]["label_tracks"] = [exact_track, alternate_track]
    with pytest.raises(CatalogValidationError, match="valid approved catalog task input"):
        resolve_catalog_task_manifest_songs(manifest, tmp_path)


def test_rejects_tampered_preprocessing_and_catalog_lineage(tmp_path: Path) -> None:
    _catalog(tmp_path, [_record(tmp_path, "octave-src-aaaaaaaa")])
    manifest = build_catalog_task_manifest(tmp_path, "section_guitar")

    manifest["task"]["preprocessing"] = {"sample_rate": 44100}
    with pytest.raises(CatalogValidationError, match="preprocessing lineage"):
        resolve_catalog_task_manifest_songs(manifest, tmp_path)

    manifest = build_catalog_task_manifest(tmp_path, "fret_mapper_bass")
    (tmp_path / "records.jsonl").write_text((tmp_path / "records.jsonl").read_text() + "\n")
    with pytest.raises(CatalogValidationError, match="catalog lineage"):
        resolve_catalog_task_manifest_songs(manifest, tmp_path)


def test_runtime_rechecks_allowed_rights_and_asset_hashes(tmp_path: Path) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa")
    _catalog(tmp_path, [record])
    manifest = build_catalog_task_manifest(tmp_path, "vocals")

    record["rights"]["training_use"] = "review_required"
    _catalog(tmp_path, [record])
    with pytest.raises(CatalogValidationError, match="catalog lineage"):
        resolve_catalog_task_manifest_songs(manifest, tmp_path)
