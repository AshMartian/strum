import hashlib
import io
import json
import struct
import sys
import wave
from pathlib import Path

import pytest

# ruff: noqa: E402
mido = pytest.importorskip("mido")

from scripts.prepare_guitar_chart_pairs import PreparationError
from scripts.train_chart_transform import TrainingConfig, load_dataset, train
from src.catalog_chart_pairs import (
    CatalogChartPairOptions,
    pipeline_descriptor,
    prepare_catalog_chart_pairs,
)
from src.model_bundle import MANIFEST_FILENAME
from src.worker import WorkerRequestError, main, preflight_bundle, run_training_request


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
        "media_type": "audio/midi" if filename.endswith(".mid") else "audio/wav",
    }


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(struct.pack("<" + "h" * 32_000, *([0] * 32_000)))
    return output.getvalue()


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
            _record(tmp_path, "octave-src-33333333", track_name, 2),
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

    assert result["record_count"] == 3
    assert manifest["task_view"]["pipeline"] == {
        "id": "chart_transform.five_lane",
        "version": 1,
    }
    assert manifest["task_view"]["catalog"]["catalog_id"] == "fixture-catalog"
    assert {record["source_id"] for record in records} == {
        "octave-src-11111111",
        "octave-src-22222222",
        "octave-src-33333333",
    }
    assert {record["split"] for record in records} == {"train", "calibration", "test"}
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
    assert len(pairs) == 3
    assert loaded_manifest["task_view"]["task_view_id"] == result["task_view_id"]
    training = train(config)
    assert training["metadata"]["dataset"]["task_view"]["task_view_id"] == result["task_view_id"]
    assert training["metadata"]["split"]["algorithm"] == "sha256-source-id-rank/v2-three-way"


def test_rejects_pair_that_no_longer_matches_catalog_task_view(tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        [
            _record(tmp_path, "octave-src-11111111", "PART GUITAR", 0),
            _record(tmp_path, "octave-src-22222222", "PART GUITAR", 1),
            _record(tmp_path, "octave-src-33333333", "PART GUITAR", 2),
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


def test_new_catalog_preparation_requires_three_source_disjoint_records(tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        [
            _record(tmp_path, "octave-src-11111111", "PART GUITAR", 0),
            _record(tmp_path, "octave-src-22222222", "PART GUITAR", 1),
        ],
    )
    with pytest.raises(PreparationError, match="at least three valid song records"):
        prepare_catalog_chart_pairs(
            tmp_path,
            tmp_path / "task-view",
            CatalogChartPairOptions(instrument="guitar", target_difficulty="Hard"),
        )


@pytest.mark.parametrize("field", ["split_seed", "calibration_fraction", "test_fraction"])
def test_catalog_three_way_split_options_reject_booleans(field: str) -> None:
    values: dict[str, object] = {
        "instrument": "guitar",
        "target_difficulty": "Hard",
        field: True,
    }
    with pytest.raises(PreparationError, match="split_seed|calibration_fraction|test_fraction"):
        CatalogChartPairOptions(**values)  # type: ignore[arg-type]


def test_worker_trains_audio_conditioned_transform_from_private_catalog_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        _record(tmp_path, "octave-src-11111111", "PART GUITAR", 0),
        _record(tmp_path, "octave-src-22222222", "PART GUITAR", 1),
        _record(tmp_path, "octave-src-33333333", "PART GUITAR", 2),
    ]
    for record in records:
        record["audio"] = {"guitar": _asset(tmp_path, _wav_bytes(), "guitar.wav")}
    _catalog(tmp_path, records)
    prepared = prepare_catalog_chart_pairs(
        tmp_path,
        tmp_path / "task-view",
        CatalogChartPairOptions(
            instrument="guitar",
            target_difficulty="Hard",
            audio_feature_mode="rms_onset_v1",
        ),
    )
    manifest_path = prepared["manifest_path"]
    assert isinstance(manifest_path, Path)
    monkeypatch.setattr(
        "scripts.train_chart_transform.event_audio_features",
        lambda _path, events, **_kwargs: [[0.0, 0.0] for _ in events],
    )
    request = tmp_path / "audio-conditioned-request.json"
    request.write_text(
        json.dumps(
            {
                "pipeline_id": "chart_transform.five_lane/v1",
                "task_view": str(manifest_path),
                "output": str(tmp_path / "audio-conditioned-experiment"),
                "catalog_root": str(tmp_path),
                "options": {
                    "model_id": "audio-conditioned-catalog-transform",
                    "epochs": 1,
                    "hidden_dim": 4,
                    "device": "cpu",
                },
            }
        )
    )

    result = run_training_request(request)

    output = tmp_path / "audio-conditioned-experiment"
    metadata = json.loads((output / "training-metadata.json").read_text())
    experiment = json.loads((output / "experiment.json").read_text())
    model_config = json.loads((output / "configs" / "training-config.json").read_text())
    assert result["status"] == "completed"
    assert metadata["audio_conditioning"]["mode"] == "rms_onset_v1"
    catalog_audio = metadata["audio_conditioning"]["catalog"]
    assert catalog_audio["format"] == "strum-catalog-audio-conditioning/v1"
    assert [asset["audio_role"] for asset in catalog_audio["assets"]] == ["guitar"] * 3
    assert experiment["catalog_audio_conditioning"] == catalog_audio
    assert model_config["audio_manifest"] is None
    assert str(tmp_path) not in json.dumps({"metadata": metadata, "experiment": experiment})
    assert not list(tmp_path.glob(".strum-chart-audio-*"))


def test_worker_promotes_audio_transform_from_private_catalog_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    request: pytest.FixtureRequest,
) -> None:
    """Audio candidates can be admitted without retaining their train scratch manifest."""
    # This verifies catalog-private audio materialization, not convergence on
    # the deliberately tiny fixture. Keep promotion evidence otherwise valid.
    monkeypatch.setattr(
        "src.chart_transform_profile.metrics_from_probabilities",
        lambda *_: {"lane_precision": 0.8, "lane_recall": 0.8, "lane_f1": 0.8},
    )
    records = [
        _record(tmp_path, "octave-src-11111111", "PART GUITAR", 0),
        _record(tmp_path, "octave-src-22222222", "PART GUITAR", 1),
        _record(tmp_path, "octave-src-33333333", "PART GUITAR", 2),
    ]
    for record in records:
        record["audio"] = {"guitar": _asset(tmp_path, _wav_bytes(), "guitar.wav")}
    _catalog(tmp_path, records)
    prepared = prepare_catalog_chart_pairs(
        tmp_path,
        tmp_path / "task-view",
        CatalogChartPairOptions(
            instrument="guitar",
            target_difficulty="Hard",
            split_seed=11,
            audio_feature_mode="rms_onset_v1",
        ),
    )
    manifest_path = prepared["manifest_path"]
    assert isinstance(manifest_path, Path)
    monkeypatch.setattr(
        "scripts.train_chart_transform.event_audio_features",
        lambda _path, events, **_kwargs: [[0.0, 0.0] for _ in events],
    )
    training_request = tmp_path / "audio-conditioned-request.json"
    candidate_parent = tmp_path / "candidate-store"
    candidate = candidate_parent / "audio-conditioned-experiment"
    training_request.write_text(
        json.dumps(
            {
                "pipeline_id": "chart_transform.five_lane/v1",
                "task_view": str(manifest_path),
                "output": str(candidate),
                "catalog_root": str(tmp_path),
                "options": {
                    "model_id": "audio-conditioned-catalog-transform",
                    "epochs": 1,
                    "hidden_dim": 4,
                    "seed": 11,
                    "device": "cpu",
                },
            }
        )
    )
    run_training_request(training_request)
    assert not list(tmp_path.glob(".strum-chart-audio-*"))
    original_mode = candidate_parent.stat().st_mode
    candidate_parent.chmod(original_mode & ~0o222)
    request.addfinalizer(lambda: candidate_parent.chmod(original_mode))

    evaluation = tmp_path / "held-out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "strum-worker",
            "transform",
            "profile",
            "evaluate",
            "--bundle-root",
            str(candidate),
            "--dataset-manifest",
            str(manifest_path),
            "--catalog-root",
            str(tmp_path),
            "--output",
            str(evaluation),
        ],
    )
    assert main() == 0
    evaluation_result = json.loads(capsys.readouterr().out)
    report = json.loads(evaluation.read_text())
    assert evaluation_result["split"] == "test"
    assert report["audio_manifest_sha256"]
    assert str(tmp_path) not in json.dumps(report)
    assert not list(tmp_path.glob(".strum-chart-audio-*"))

    profile = tmp_path / "promoted"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "strum-worker",
            "transform",
            "profile",
            "package",
            "--experiment",
            str(candidate),
            "--evaluation",
            str(evaluation),
            "--dataset-manifest",
            str(manifest_path),
            "--catalog-root",
            str(tmp_path),
            "--output",
            str(profile),
            "--profile",
            "audio-transform-guitar",
        ],
    )
    assert main() == 0
    package_result = json.loads(capsys.readouterr().out)
    assert package_result["status"] == "promoted"
    assert preflight_bundle(profile)["status"] == "ready"
    assert str(tmp_path) not in (profile / MANIFEST_FILENAME).read_text()
    assert not list(tmp_path.glob(".strum-chart-audio-*"))


def test_worker_catalog_audio_promotion_rejects_tampered_candidate_or_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        _record(tmp_path, "octave-src-11111111", "PART GUITAR", 0),
        _record(tmp_path, "octave-src-22222222", "PART GUITAR", 1),
        _record(tmp_path, "octave-src-33333333", "PART GUITAR", 2),
    ]
    for record in records:
        record["audio"] = {"guitar": _asset(tmp_path, _wav_bytes(), "guitar.wav")}
    _catalog(tmp_path, records)
    prepared = prepare_catalog_chart_pairs(
        tmp_path,
        tmp_path / "task-view",
        CatalogChartPairOptions(
            instrument="guitar",
            target_difficulty="Hard",
            audio_feature_mode="rms_onset_v1",
        ),
    )
    manifest_path = prepared["manifest_path"]
    assert isinstance(manifest_path, Path)
    monkeypatch.setattr(
        "scripts.train_chart_transform.event_audio_features",
        lambda _path, events, **_kwargs: [[0.0, 0.0] for _ in events],
    )
    request = tmp_path / "request.json"
    candidate = tmp_path / "candidate"
    request.write_text(
        json.dumps(
            {
                "pipeline_id": "chart_transform.five_lane/v1",
                "task_view": str(manifest_path),
                "output": str(candidate),
                "catalog_root": str(tmp_path),
                "options": {
                    "model_id": "audio-conditioned-catalog-transform",
                    "epochs": 1,
                    "hidden_dim": 4,
                    "device": "cpu",
                },
            }
        )
    )
    run_training_request(request)

    config = candidate / "configs" / "training-config.json"
    original_config = config.read_bytes()
    config.write_text("{}\n")
    from src.worker import evaluate_catalog_chart_transform_candidate

    with pytest.raises(WorkerRequestError, match="candidate failed verification"):
        evaluate_catalog_chart_transform_candidate(
            bundle_root=candidate,
            dataset_manifest=manifest_path,
            catalog_root=tmp_path,
            output_path=tmp_path / "held-out.json",
        )
    config.write_bytes(original_config)

    relative_audio = records[0]["audio"]["guitar"]["relative_path"]
    assert isinstance(relative_audio, str)
    (tmp_path / relative_audio).write_bytes(b"tampered")
    with pytest.raises(WorkerRequestError, match="catalog failed verification"):
        evaluate_catalog_chart_transform_candidate(
            bundle_root=candidate,
            dataset_manifest=manifest_path,
            catalog_root=tmp_path,
            output_path=tmp_path / "held-out.json",
        )


def test_pipeline_descriptor_declares_catalog_and_preprocessing_contract() -> None:
    descriptor = pipeline_descriptor()

    assert descriptor["pipeline_id"] == "chart_transform.five_lane"
    assert descriptor["required_catalog"]["training_use"] == "allowed"
    assert descriptor["split"]["algorithm"] == "sha256-source-id-rank/v2-three-way"
    assert descriptor["training_requirements"] == [
        "source_disjoint_train_calibration_test/v2",
        "strum_owned_decoder_calibration/v1",
        "test_only_transform_promotion/v1",
    ]
