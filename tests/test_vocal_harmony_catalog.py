from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import mido
import pytest

from src.catalog_task_manifest import _catalog_fingerprint
from src.song_source_catalog import CatalogValidationError, load_catalog
from src.vocal_harmony_catalog import (
    HARMONY_SOURCE_POLICY_FILENAME,
    HARMONY_SOURCE_TASK_FORMAT,
    build_vocal_harmony_source_task,
    inspect_vocal_harmony_source_catalog,
    resolve_vocal_harmony_source_task,
)
from src.worker import PIPELINES, prepare_dataset_request


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
        "media_type": "audio/wav" if filename.endswith(".wav") else "audio/midi",
    }


def _harmony_midi() -> bytes:
    midi = mido.MidiFile(ticks_per_beat=480)
    for track_name, pitch in (("PART VOCALS", 60), ("HARM1", 64)):
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=track_name, time=0))
        track.append(mido.Message("note_on", note=pitch, velocity=100, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=480))
        midi.tracks.append(track)
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue()


def _record(root: Path) -> dict[str, object]:
    notes = _asset(root, _harmony_midi(), "notes.mid")
    mix = _asset(root, b"mix-master", "mix.wav")
    harm1 = _asset(root, b"isolated-harm1", "harm1.wav")
    return {
        "source_id": "octave-src-harmony001",
        "import": {"kind": "song_folder", "adapter_version": "octave-folder-1", "warnings": []},
        "rights": {
            "training_use": "allowed",
            "provenance": "reviewed-license",
            "license": "test-only",
        },
        "metadata": {"name": "Harmony fixture"},
        "chart": {
            "notes_midi": notes,
            "instruments": {
                "vocals": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART VOCALS", "HARM1"],
                }
            },
        },
        "audio": {
            "mix": mix,
            "harm1": harm1,
            "vocals": _asset(root, b"shared-vocals", "vocals.wav"),
        },
    }


def _write_catalog(root: Path, record: dict[str, object]) -> None:
    (root / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "harmony-fixture",
                "records": "records.jsonl",
            }
        ),
        encoding="utf-8",
    )


def _write_policy(root: Path, record: dict[str, object], *, separated: bool = True) -> None:
    catalog = load_catalog(root)
    if separated:
        provenance: dict[str, object] = {
            "kind": "isolated_separation_output/v1",
            "timeline": "same-master-timeline/v1",
            "input": {
                "asset_id": record["audio"]["mix"]["asset_id"],
                "sha256": record["audio"]["mix"]["sha256"],
            },
            "separator": {
                "id": "demucs",
                "version": "v4",
                "model_sha256": "a" * 64,
                "configuration_sha256": "b" * 64,
            },
        }
    else:
        provenance = {
            "kind": "isolated_source_stem/v1",
            "timeline": "same-master-timeline/v1",
            "attestation_id": "licensed-stem-001",
        }
    policy = {
        "schema_version": 1,
        "format": "octave-vocal-harmony-source-policy/v1",
        "policy_id": "octave-harmony-fixture-1",
        "catalog_id": catalog.catalog_id,
        "catalog_control_sha256": _catalog_fingerprint(catalog),
        "records": [
            {
                "source_id": record["source_id"],
                "track_name": "HARM1",
                "audio": {
                    "role": "harm1",
                    "asset_id": record["audio"]["harm1"]["asset_id"],
                    "sha256": record["audio"]["harm1"]["sha256"],
                },
                "provenance": provenance,
            }
        ],
    }
    (root / HARMONY_SOURCE_POLICY_FILENAME).write_text(json.dumps(policy), encoding="utf-8")


def test_harmony_source_task_requires_policy_and_never_accepts_shared_vocals(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    _write_catalog(tmp_path, record)

    inspection = inspect_vocal_harmony_source_catalog(tmp_path)
    assert inspection["eligible_count"] == 0
    assert inspection["audio_policy"]["shared_vocals_or_mix_allowed"] is False
    with pytest.raises(CatalogValidationError, match="source policy is missing"):
        build_vocal_harmony_source_task(tmp_path)

    _write_policy(tmp_path, record)
    policy_path = tmp_path / HARMONY_SOURCE_POLICY_FILENAME
    policy = json.loads(policy_path.read_text())
    policy["records"][0]["audio"]["role"] = "vocals"
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(CatalogValidationError, match="audio role does not match HARM track"):
        build_vocal_harmony_source_task(tmp_path)


def test_harmony_source_task_binds_exact_tracks_assets_provenance_and_catalog(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    _write_catalog(tmp_path, record)
    _write_policy(tmp_path, record)

    task = build_vocal_harmony_source_task(
        tmp_path, harmony_tracks=["HARM1"], split_ratios=(50, 50, 0), split_seed="harmony-smoke"
    )
    assert task["format"] == HARMONY_SOURCE_TASK_FORMAT
    assert task["summary"] == {
        "record_count": 1,
        "source_count": 1,
        "by_split": {task["sources"][0]["split"]: 1},
    }
    source = task["sources"][0]
    assert source["track_name"] == "HARM1"
    assert source["audio_role"] == "harm1"
    assert source["provenance"]["kind"] == "isolated_separation_output/v1"
    assert source["provenance"]["separator"]["model_sha256"] == "a" * 64
    rendered = json.dumps(task)
    assert str(tmp_path) not in rendered
    assert "shared-vocals" not in rendered

    resolved = resolve_vocal_harmony_source_task(task, tmp_path)
    assert resolved[0]["audio_role"] == "harm1"
    assert resolved[0]["audio_path"].endswith("harm1.wav")

    task["sources"][0]["audio_role"] = "vocals"
    with pytest.raises(
        CatalogValidationError, match="does not match current approved catalog policy"
    ):
        resolve_vocal_harmony_source_task(task, tmp_path)


def test_harmony_policy_requires_pinned_separation_configuration(tmp_path: Path) -> None:
    record = _record(tmp_path)
    _write_catalog(tmp_path, record)
    _write_policy(tmp_path, record)
    policy_path = tmp_path / HARMONY_SOURCE_POLICY_FILENAME
    policy = json.loads(policy_path.read_text())
    del policy["records"][0]["provenance"]["separator"]["configuration_sha256"]
    policy_path.write_text(json.dumps(policy))

    with pytest.raises(CatalogValidationError, match="pinned separator metadata"):
        build_vocal_harmony_source_task(tmp_path)


def test_worker_exposes_and_prepares_only_harmony_source_policy(tmp_path: Path) -> None:
    record = _record(tmp_path)
    _write_catalog(tmp_path, record)
    _write_policy(tmp_path, record, separated=False)
    descriptor = next(item for item in PIPELINES if item.id == "vocals.harmony-source-policy/v1")
    assert descriptor.preparation_status == "available"
    assert descriptor.training_status == "not_available"
    assert descriptor.train_schema is None
    assert descriptor.inference_capability is None
    assert descriptor.catalog_requirements["audio_policy"] == (
        "isolated_harmony_only:no_fallback_to_vocals_or_mix"
    )

    output = tmp_path / "views" / "harmony.json"
    request = tmp_path / "prepare.json"
    request.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": descriptor.id,
                "output": str(output),
                "options": {"harmony_tracks": ["HARM1"]},
            }
        )
    )
    result = prepare_dataset_request(request)
    assert result["status"] == "prepared"
    assert result["record_count"] == 1
    assert str(tmp_path) not in output.read_text()
