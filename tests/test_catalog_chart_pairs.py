import hashlib
import io
import json
from pathlib import Path

import pytest

# ruff: noqa: E402
mido = pytest.importorskip("mido")

from scripts.train_chart_transform import TrainingConfig, load_dataset, train
from src.catalog_chart_pairs import (
    CatalogChartPairOptions,
    pipeline_descriptor,
    prepare_catalog_chart_pairs,
)


def _midi_bytes(track_name: str, variation: int) -> bytes:
    midi = mido.MidiFile(ticks_per_beat=480)
    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    midi.tracks.append(tempo)
    instrument = mido.MidiTrack()
    instrument.append(mido.MetaMessage("track_name", name=track_name, time=0))
    instrument.append(mido.Message("note_on", note=96, velocity=100, time=0))
    instrument.append(mido.Message("note_on", note=84, velocity=100, time=240))
    instrument.append(mido.Message("note_on", note=97 + variation % 2, velocity=100, time=240))
    midi.tracks.append(instrument)
    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()


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
        "media_type": "audio/midi",
    }


def _record(
    root: Path, source_id: str, track_name: str, variation: int, rights: str = "allowed"
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "import": {"kind": "song_folder", "adapter_version": "octave-folder/1", "warnings": []},
        "rights": {"training_use": rights, "provenance": "Reviewed", "license": "test-only"},
        "metadata": {"name": "Fixture", "artist": "STRUM"},
        "chart": {
            "notes_midi": _asset(root, _midi_bytes(track_name, variation), "notes.mid"),
            "instruments": {
                _instrument_for_track(track_name): {
                    "status": "present",
                    "difficulties": ["expert", "hard"],
                    "track_names": [track_name],
                }
            },
        },
        "audio": {},
    }


def _instrument_for_track(track_name: str) -> str:
    return {
        "PART GUITAR": "guitar",
        "PART BASS": "bass",
        "PART KEYS": "keys",
        "PART DRUMS": "drums",
    }[track_name]


def _catalog(root: Path, records: list[dict[str, object]]) -> None:
    (root / "records.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "fixture-catalog",
                "records": "records.jsonl",
            }
        )
    )


@pytest.mark.parametrize(
    ("instrument", "track_name"),
    [
        ("guitar", "PART GUITAR"),
        ("bass", "PART BASS"),
        ("keys", "PART KEYS"),
        ("drums", "PART DRUMS"),
    ],
)
def test_builds_path_free_catalog_task_view_for_each_five_lane_instrument(
    tmp_path: Path, instrument: str, track_name: str
) -> None:
    _catalog(
        tmp_path,
        [
            _record(tmp_path, "octave-src-11111111", track_name, 0),
            _record(tmp_path, "octave-src-22222222", track_name, 1),
            _record(tmp_path, "octave-src-33333333", track_name, 2, rights="review_required"),
        ],
    )

    result = prepare_catalog_chart_pairs(
        tmp_path,
        tmp_path / "task-view",
        CatalogChartPairOptions(instrument=instrument, target_difficulty="Hard", split_seed=17),
    )
    manifest_path = result["manifest_path"]
    assert isinstance(manifest_path, Path)
    manifest = json.loads(manifest_path.read_text())
    records = [
        json.loads(line) for line in (manifest_path.parent / "pairs.jsonl").read_text().splitlines()
    ]

    assert result["record_count"] == 2
    assert manifest["task_view"]["pipeline"] == {
        "id": "chart_transform.five_lane",
        "version": 1,
    }
    assert manifest["task_view"]["catalog"]["catalog_id"] == "fixture-catalog"
    assert {record["source_id"] for record in records} == {
        "octave-src-11111111",
        "octave-src-22222222",
    }
    assert {record["split"] for record in records} == {"train", "validation"}
    serialized = json.dumps({"manifest": manifest, "records": records})
    assert str(tmp_path) not in serialized

    config = TrainingConfig(
        dataset_manifest=str(manifest_path),
        output_dir=str(tmp_path / "experiment"),
        model_id="fixture-model",
        source_difficulty="Expert",
        target_difficulty="Hard",
        hidden_dim=4,
        epochs=1,
        device="cpu",
    )
    pairs, loaded_manifest = load_dataset(config)
    assert len(pairs) == 2
    assert loaded_manifest["task_view"]["task_view_id"] == result["task_view_id"]
    training = train(config)
    assert training["metadata"]["dataset"]["task_view"]["task_view_id"] == result["task_view_id"]
    assert training["metadata"]["split"]["algorithm"] == "sha256-source-id-rank/v1"


def test_rejects_pair_that_no_longer_matches_catalog_task_view(tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        [
            _record(tmp_path, "octave-src-11111111", "PART GUITAR", 0),
            _record(tmp_path, "octave-src-22222222", "PART GUITAR", 1),
        ],
    )
    result = prepare_catalog_chart_pairs(
        tmp_path,
        tmp_path / "task-view",
        CatalogChartPairOptions(instrument="guitar", target_difficulty="Hard"),
    )
    manifest_path = result["manifest_path"]
    assert isinstance(manifest_path, Path)
    pairs_path = manifest_path.parent / "pairs.jsonl"
    records = [json.loads(line) for line in pairs_path.read_text().splitlines()]
    records[0]["notes_midi_sha256"] = "0" * 64
    pairs_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    with pytest.raises(ValueError, match="does not match task_view lineage"):
        load_dataset(
            TrainingConfig(
                dataset_manifest=str(manifest_path),
                output_dir=str(tmp_path / "experiment"),
                model_id="fixture-model",
                source_difficulty="Expert",
                target_difficulty="Hard",
                device="cpu",
            )
        )


def test_pipeline_descriptor_declares_catalog_and_preprocessing_contract() -> None:
    descriptor = pipeline_descriptor()

    assert descriptor["pipeline_id"] == "chart_transform.five_lane"
    assert descriptor["required_catalog"]["training_use"] == "allowed"
    assert descriptor["split"]["algorithm"] == "sha256-source-id-rank/v1"
