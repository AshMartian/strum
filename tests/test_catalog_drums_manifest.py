import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

from src.catalog_drums_manifest import (
    MANIFEST_FORMAT,
    build_drums_manifest,
    resolve_drums_manifest_songs,
    task_view_sha256,
)
from src.song_source_catalog import CatalogValidationError


def _asset(root: Path, content: bytes, filename: str) -> dict[str, object]:
    digest = hashlib.sha256(content).hexdigest()
    path = root / "assets" / "sha256" / digest / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "asset_id": f"sha256:{digest}",
        "sha256": digest,
        "relative_path": path.relative_to(root).as_posix(),
        "byte_length": len(content),
        "media_type": "audio/midi" if filename.endswith(".mid") else "audio/ogg",
    }


def _midi_bytes() -> bytes:
    mid = mido.MidiFile(ticks_per_beat=480)
    tempo = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500_000, time=0)])
    drums = mido.MidiTrack(
        [
            mido.MetaMessage("track_name", name="PART DRUMS", time=0),
            mido.Message("note_on", note=96, velocity=100, time=480),
            mido.Message("note_on", note=98, velocity=100, time=0),
            mido.Message("note_on", note=110, velocity=100, time=0),
        ]
    )
    mid.tracks.extend([tempo, drums])
    output = io.BytesIO()
    mid.save(file=output)
    return output.getvalue()


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    sf.write(output, np.zeros(44_100 * 2, dtype=np.float32), 44_100, format="WAV")
    return output.getvalue()


def _record(
    root: Path, source_id: str, *, roles: tuple[str, ...], allowed: str = "allowed"
) -> dict:
    return {
        "source_id": source_id,
        "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
        "rights": {"training_use": allowed, "provenance": "Reviewed", "license": "test"},
        "metadata": {"name": "Drum song"},
        "chart": {
            "notes_midi": _asset(root, _midi_bytes(), f"{source_id}.mid"),
            "instruments": {
                "drums": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART DRUMS"],
                }
            },
        },
        "audio": {
            role: _asset(root, f"{source_id}-{role}".encode(), f"{role}.ogg") for role in roles
        },
    }


def _catalog(root: Path, records: list[dict]) -> None:
    (root / "records.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "drums-test",
                "records": "records.jsonl",
            }
        )
    )


def test_builds_portable_drums_task_view_with_lineage(tmp_path: Path) -> None:
    fallback = _record(tmp_path, "octave-src-aaaaaaaa", roles=("mix",))
    preferred = _record(tmp_path, "octave-src-bbbbbbbb", roles=("drums", "mix"))
    excluded = _record(tmp_path, "octave-src-cccccccc", roles=("drums",), allowed="review_required")
    _catalog(tmp_path, [preferred, excluded, fallback])

    manifest = build_drums_manifest(tmp_path)
    serialized = json.dumps(manifest)

    assert manifest["format"] == MANIFEST_FORMAT
    assert [song["audio_role"] for song in manifest["songs"]] == ["mix", "drums"]
    assert manifest["task"]["pipeline_id"] == "drums.onset-classifier"
    assert "runtime_admission" not in manifest["task"]
    assert "runtime_admission" not in manifest["summary"]
    assert manifest["catalog"]["content_sha256"]
    assert str(tmp_path) not in serialized
    assert "provenance" not in serialized
    assert task_view_sha256(manifest) == task_view_sha256(manifest)

    resolved = resolve_drums_manifest_songs(manifest, tmp_path)
    assert [song["source_id"] for song in resolved] == [
        "octave-src-aaaaaaaa",
        "octave-src-bbbbbbbb",
    ]
    assert all("audio_sha256" in song["input_hashes"] for song in resolved)


def test_drums_manifest_rejects_five_lane_runtime_admission_marker(tmp_path: Path) -> None:
    _catalog(tmp_path, [_record(tmp_path, "octave-src-aaaaaaaa", roles=("drums",))])

    with pytest.raises(TypeError, match="runtime_admission"):
        build_drums_manifest(tmp_path, runtime_admission=True)  # type: ignore[call-arg]

    manifest = build_drums_manifest(tmp_path)
    manifest["task"]["runtime_admission"] = "strum-five-lane-runtime-admission/v1"
    with pytest.raises(CatalogValidationError, match="manifest task is invalid"):
        resolve_drums_manifest_songs(manifest, tmp_path)


def test_rejects_catalog_changes_and_asset_tampering(tmp_path: Path) -> None:
    _catalog(tmp_path, [_record(tmp_path, "octave-src-aaaaaaaa", roles=("drums",))])
    manifest = build_drums_manifest(tmp_path)
    manifest["songs"][0]["audio"]["relative_path"] = "assets/sha256/" + "0" * 64 + "/drums.ogg"
    with pytest.raises(CatalogValidationError, match="approved Drums"):
        resolve_drums_manifest_songs(manifest, tmp_path)

    manifest = build_drums_manifest(tmp_path)
    changed = json.loads((tmp_path / "records.jsonl").read_text())
    changed["chart"]["instruments"]["drums"]["difficulties"] = ["hard"]
    (tmp_path / "records.jsonl").write_text(json.dumps(changed) + "\n")
    with pytest.raises(CatalogValidationError, match="catalog content"):
        resolve_drums_manifest_songs(manifest, tmp_path)


def test_catalog_preprocessing_writes_path_free_lineage(tmp_path: Path) -> None:
    record = _record(tmp_path, "octave-src-aaaaaaaa", roles=("drums",))
    record["audio"]["drums"] = _asset(tmp_path, _wav_bytes(), "drums.wav")
    _catalog(tmp_path, [record])
    manifest_path = tmp_path / "view.json"
    manifest_path.write_text(json.dumps(build_drums_manifest(tmp_path)))
    split = json.loads(manifest_path.read_text())["songs"][0]["split"]
    output = tmp_path / "cache-output"
    script = Path(__file__).parents[1] / "scripts" / "preprocess_onset_windows.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--catalog-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--split",
            split,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    index = json.loads((output / "cache" / f"{split}_index.json").read_text())
    lineage = index["lineage"]
    assert lineage["catalog_id"] == "drums-test"
    assert lineage["pipeline_id"] == "drums.onset-classifier"
    assert lineage["source_ids"] == ["octave-src-aaaaaaaa"]
    assert str(tmp_path) not in json.dumps(lineage)
    labels = np.load(output / "cache" / f"{split}_labels.npy")
    assert labels.shape == (1, 8)
    assert labels[0, 0] == 1  # kick
    assert labels[0, 3] == 1  # yellow with a 110 marker -> high tom
