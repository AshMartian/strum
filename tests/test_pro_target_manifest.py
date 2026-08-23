from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import mido
import pytest

from src.catalog_task_manifest import build_catalog_task_manifest
from src.pro_target_manifest import (
    PRO_TARGET_MANIFEST_FORMAT,
    build_catalog_pro_target_manifest,
    resolve_catalog_pro_target_manifest_songs,
)
from src.song_source_catalog import CatalogValidationError
from src.worker import prepare_dataset_request


def _asset(root: Path, content: bytes, filename: str) -> dict[str, object]:
    digest = hashlib.sha256(content).hexdigest()
    relative = f"assets/sha256/{digest}/{filename}"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "asset_id": f"sha256:{digest}",
        "sha256": digest,
        "relative_path": relative,
        "byte_length": len(content),
        "media_type": "audio/midi" if filename.endswith(".mid") else "audio/ogg",
    }


def _track(name: str, messages: list[mido.Message]) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    track.extend(messages)
    return track


def _pro_midi(*, standard_fret: int = 3) -> bytes:
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(
        _track(
            "PART REAL_GUITAR",
            [
                mido.Message("note_on", note=96, velocity=100 + standard_fret, channel=4, time=12),
                mido.Message("note_off", note=96, velocity=0, channel=4, time=120),
            ],
        )
    )
    midi.tracks.append(
        _track(
            "PART REAL_GUITAR_22",
            [
                mido.Message("note_on", note=101, velocity=122, channel=3, time=24),
                mido.Message("note_off", note=101, velocity=0, channel=3, time=60),
            ],
        )
    )
    midi.tracks.append(
        _track(
            "PART REAL_BASS_22",
            [
                mido.Message("note_on", note=98, velocity=115, channel=0, time=36),
                mido.Message("note_off", note=98, velocity=0, channel=0, time=48),
            ],
        )
    )
    midi.tracks.append(
        _track(
            "PART REAL_KEYS_X",
            [
                mido.Message("note_on", note=0, velocity=1, time=4),
                mido.Message("note_off", note=0, velocity=0, time=0),
                mido.Message("note_on", note=60, velocity=100, channel=1, time=8),
                mido.Message("note_off", note=60, velocity=0, channel=1, time=240),
            ],
        )
    )
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue()


def _catalog(root: Path, *, standard_fret: int = 3) -> None:
    source_id = "octave-src-pro-targets-0001"
    record = {
        "source_id": source_id,
        "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
        "rights": {"training_use": "allowed", "provenance": "reviewed", "license": "test-only"},
        "metadata": {"name": "Pro target test"},
        "chart": {
            "notes_midi": _asset(root, _pro_midi(standard_fret=standard_fret), "notes.mid"),
            "instruments": {
                "pro_guitar": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART REAL_GUITAR", "PART REAL_GUITAR_22"],
                },
                "pro_bass": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART REAL_BASS_22"],
                },
                "pro_keys": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART REAL_KEYS_X"],
                },
            },
        },
        "audio": {
            role: _asset(root, f"{source_id}:{role}".encode(), f"{role}.ogg")
            for role in ("mix", "guitar", "bass", "keys")
        },
    }
    (root / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "pro-target-test",
                "records": "records.jsonl",
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("task_kind", "expected_schema", "expected_event"),
    [
        (
            "pro_guitar",
            "pro-string-fret-events/v1",
            {"string": 0, "fret": 3, "technique": "tapped"},
        ),
        ("pro_bass", "pro-string-fret-events/v1", {"string": 2, "fret": 15, "technique": "normal"}),
        ("pro_keys", "pro-keys-pitch-events/v1", {"pitch": 60, "channel": 1}),
    ],
)
def test_pro_targets_are_decoded_from_exact_tracks_without_path_leaks(
    tmp_path: Path, task_kind: str, expected_schema: str, expected_event: dict[str, object]
) -> None:
    _catalog(tmp_path)

    manifest = build_catalog_pro_target_manifest(tmp_path, task_kind)

    serialized = json.dumps(manifest)
    assert manifest["format"] == PRO_TARGET_MANIFEST_FORMAT
    assert manifest["target_encoding"]["id"] == "strum-pro-midi-target-decoder/v1"
    assert str(tmp_path) not in serialized
    track = manifest["songs"][0]["targets"][0]
    assert track["event_schema"] == expected_schema
    assert expected_event.items() <= track["events"][0].items()
    if task_kind == "pro_guitar":
        assert manifest["songs"][0]["targets"][1]["track_variant"] == "22_fret"
    if task_kind == "pro_keys":
        assert track["range_shifts"] == [{"tick": 4, "anchor": "C"}]

    resolved = resolve_catalog_pro_target_manifest_songs(manifest, tmp_path)
    assert str(tmp_path) in resolved[0]["midi_path"]
    assert resolved[0]["targets"] == manifest["songs"][0]["targets"]


def test_pro_targets_exclude_invalid_standard_fret_and_remain_non_trainable(tmp_path: Path) -> None:
    _catalog(tmp_path, standard_fret=22)

    manifest = build_catalog_pro_target_manifest(tmp_path, "pro_guitar")

    assert manifest["songs"] == []
    assert manifest["summary"]["coverage_record_count"] == 1
    assert manifest["summary"]["exclusion_reason_counts"] == {
        "Pro string target uses an unsupported technique or fret": 1
    }
    # The standard task contract continues to advertise no trainer; decoding
    # labels is not a claim that STRUM can generate playable Pro charts.
    assert build_catalog_task_manifest(tmp_path, "pro_guitar")["task"]["kind"] == "pro_guitar"


def test_pro_target_resolution_rejects_tampering_or_catalog_drift(tmp_path: Path) -> None:
    _catalog(tmp_path)
    manifest = build_catalog_pro_target_manifest(tmp_path, "pro_keys")
    manifest["songs"][0]["targets"][0]["events"][0]["pitch"] = 61
    with pytest.raises(CatalogValidationError, match="targets do not match"):
        resolve_catalog_pro_target_manifest_songs(manifest, tmp_path)


def test_worker_prepare_returns_a_decoded_pro_target_view(tmp_path: Path) -> None:
    _catalog(tmp_path)
    output = tmp_path / "out" / "pro-keys-targets.json"
    request = tmp_path / "prepare.json"
    request.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": "strum.instrument-chart/pro-keys/v1",
                "output": str(output),
                "options": {},
            }
        ),
        encoding="utf-8",
    )

    result = prepare_dataset_request(request)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert result["status"] == "prepared"
    assert result["record_count"] == 1
    assert result["output_name"] == output.name
    assert written["format"] == PRO_TARGET_MANIFEST_FORMAT
    assert str(tmp_path) not in json.dumps(written)
