import hashlib
import json
from pathlib import Path

import pytest

from src.catalog_guitar_manifest import (
    MANIFEST_FORMAT,
    build_guitar_manifest,
    resolve_guitar_manifest_songs,
    write_guitar_manifest,
)
from src.song_source_catalog import CatalogValidationError


def _asset(root: Path, content: bytes, filename: str) -> dict[str, object]:
    sha256 = hashlib.sha256(content).hexdigest()
    path = root / "assets" / "sha256" / sha256 / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "asset_id": f"sha256:{sha256}",
        "sha256": sha256,
        "relative_path": path.relative_to(root).as_posix(),
        "byte_length": len(content),
        "media_type": "audio/midi" if filename.endswith(".mid") else "audio/ogg",
    }


def _record(
    root: Path,
    source_id: str,
    *,
    training_use: str = "allowed",
    roles: tuple[str, ...] = ("guitar",),
) -> dict[str, object]:
    audio = {role: _asset(root, f"{source_id}-{role}".encode(), f"{role}.ogg") for role in roles}
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
            "instruments": {
                "guitar": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART GUITAR"],
                }
            },
        },
        "audio": audio,
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


def test_builds_path_free_manifest_and_prefers_guitar_audio(tmp_path: Path) -> None:
    fallback_only = _record(tmp_path, "octave-src-aaaaaaaa", roles=("mix",))
    preferred = _record(tmp_path, "octave-src-bbbbbbbb", roles=("guitar", "mix"))
    review_required = _record(
        tmp_path, "octave-src-cccccccc", training_use="review_required", roles=("guitar",)
    )
    _catalog(tmp_path, [preferred, review_required, fallback_only])

    manifest = build_guitar_manifest(tmp_path)
    serialized = json.dumps(manifest)

    assert manifest["format"] == MANIFEST_FORMAT
    assert [song["source_id"] for song in manifest["songs"]] == [
        "octave-src-aaaaaaaa",
        "octave-src-bbbbbbbb",
    ]
    assert [song["audio_role"] for song in manifest["songs"]] == ["mix", "guitar"]
    assert str(tmp_path) not in serialized
    assert "provenance" not in serialized

    output = write_guitar_manifest(tmp_path / "views" / "guitar.json", manifest)
    assert json.loads(output.read_text()) == manifest

    resolved = resolve_guitar_manifest_songs(manifest, tmp_path)
    assert [song["source_id"] for song in resolved] == [
        "octave-src-aaaaaaaa",
        "octave-src-bbbbbbbb",
    ]
    assert all(str(tmp_path) in song["audio_path"] for song in resolved)


def test_rejects_tampered_task_asset_reference(tmp_path: Path) -> None:
    _catalog(tmp_path, [_record(tmp_path, "octave-src-aaaaaaaa", roles=("guitar",))])
    manifest = build_guitar_manifest(tmp_path)
    song = manifest["songs"][0]
    song["audio"]["relative_path"] = "assets/sha256/" + "0" * 64 + "/guitar.ogg"

    with pytest.raises(CatalogValidationError, match="manifest asset"):
        resolve_guitar_manifest_songs(manifest, tmp_path)


def test_rejects_tampered_split_assignment(tmp_path: Path) -> None:
    _catalog(tmp_path, [_record(tmp_path, "octave-src-aaaaaaaa", roles=("guitar",))])
    manifest = build_guitar_manifest(tmp_path)
    original_split = manifest["songs"][0]["split"]
    manifest["songs"][0]["split"] = "val" if original_split != "val" else "train"

    with pytest.raises(CatalogValidationError, match="guitar coverage"):
        resolve_guitar_manifest_songs(manifest, tmp_path)


def test_split_assignments_are_deterministic(tmp_path: Path) -> None:
    _catalog(tmp_path, [_record(tmp_path, "octave-src-aaaaaaaa", roles=("guitar",))])

    assert (
        build_guitar_manifest(tmp_path)["songs"][0]["split"]
        == build_guitar_manifest(tmp_path)["songs"][0]["split"]
    )
