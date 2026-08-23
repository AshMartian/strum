from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.catalog_task_manifest import (
    MANIFEST_FORMAT,
    PIPELINE_IDS,
    available_task_kinds,
    build_catalog_task_manifest,
    resolve_catalog_task_manifest_songs,
)
from src.song_source_catalog import CatalogValidationError


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
            "notes_midi": _asset(root, f"{source_id}-midi".encode(), "notes.mid"),
            "instruments": instruments,
        },
        "audio": {
            role: _asset(root, f"{source_id}-{role}".encode(), f"{role}.ogg")
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
