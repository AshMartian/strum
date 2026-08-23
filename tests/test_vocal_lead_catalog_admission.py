from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import mido
import pytest

from src.catalog_task_manifest import build_catalog_task_manifest
from src.vocal_lead_catalog_admission import (
    ADMISSION_FORMAT,
    RESOLVER_ID,
    VocalLeadCatalogAdmissionError,
    resolve_vocal_lead_catalog_admission,
)

_TASKS = (
    "vocals_activity",
    "vocals_phrase_boundaries",
    "vocals_lyric_alignment",
    "vocals_talky_activity",
)
_SEED = "lead-vocal-admission-test/v1"


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


def _lead_vocals_midi(*, include_talkies: bool = True) -> bytes:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="PART VOCALS", time=0))
    for index in range(50):
        track.append(mido.Message("note_on", note=60, velocity=100, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=12))
        track.append(mido.MetaMessage("lyrics", text=f"word-{index}", time=0))
        track.append(mido.Message("note_on", note=105, velocity=100, time=0))
        track.append(mido.Message("note_off", note=105, velocity=0, time=12))
        if include_talkies and index < 25:
            track.append(mido.Message("note_on", note=96, velocity=100, time=0))
            track.append(mido.Message("note_off", note=96, velocity=0, time=12))
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue()


def _record(root: Path, source_id: str, *, include_talkies: bool) -> dict[str, object]:
    return {
        "source_id": source_id,
        "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
        "rights": {
            "training_use": "allowed",
            "provenance": "Reviewed local collection",
            "license": "test-only",
        },
        "metadata": {"name": "Safe Song"},
        "chart": {
            "notes_midi": _asset(
                root,
                _lead_vocals_midi(include_talkies=include_talkies),
                "notes.mid",
            ),
            "instruments": {
                "vocals": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART VOCALS"],
                }
            },
        },
        "audio": {"vocals": _asset(root, b"not-decoded-by-admission", "vocals.ogg")},
    }


def _write_catalog(root: Path, records: list[dict[str, object]]) -> None:
    (root / "records.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "lead-admission-test",
                "records": "records.jsonl",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _source_ids_by_split(counts: dict[str, int]) -> list[str]:
    from src.catalog_task_manifest import deterministic_split

    selected: list[str] = []
    remaining = dict(counts)
    for index in range(1_000_000):
        source_id = f"octave-src-{index:08x}"
        split = deterministic_split(source_id, seed=_SEED)
        if remaining.get(split, 0):
            selected.append(source_id)
            remaining[split] -= 1
            if not any(remaining.values()):
                return selected
    raise AssertionError("could not create deterministic test source partition")


def _task_views(root: Path) -> dict[str, Path]:
    views: dict[str, Path] = {}
    for task_kind in _TASKS:
        path = root / "views" / f"{task_kind}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps(
                build_catalog_task_manifest(root, task_kind, split_seed=_SEED),
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        views[task_kind] = path
    return views


def test_catalog_resolver_recomputes_a_path_free_passing_lead_data_admission(
    tmp_path: Path,
) -> None:
    _write_catalog(
        tmp_path,
        [
            _record(tmp_path, source_id, include_talkies=True)
            for source_id in _source_ids_by_split({"train": 40, "val": 10, "test": 10})
        ],
    )

    report = resolve_vocal_lead_catalog_admission(
        catalog_root=tmp_path, task_view_paths=_task_views(tmp_path)
    )

    assert report["format"] == ADMISSION_FORMAT
    assert report["resolver"] == RESOLVER_ID
    assert report["status"] == "admitted_for_future_lead_evaluation_only"
    assert report["admission"] == {
        "passed": True,
        "scope": "lead-catalog-data-only/v1",
        "next_stage": "strum-recomputed-lead-held-out-evaluator/v1",
        "profile_packaging": "forbidden-until-full-vocal-profile-contract/v1",
        "chart_execution": "not_available",
    }
    assert report["data_coverage"]["source_counts"] == {"train": 40, "val": 10, "test": 10}
    assert report["data_coverage"]["label_counts"]["train"] == {
        "pitched_note_events": 2000,
        "phrase_boundaries": 4000,
        "lyric_events": 2000,
        "talky_spans": 1000,
    }
    assert report["label_availability"] == {
        "required": [
            "pitched_note_events",
            "phrase_boundaries",
            "lyric_events",
            "talky_spans",
        ],
        "missing": {"train": [], "val": [], "test": []},
        "passed": True,
    }
    serialized = json.dumps(report)
    assert str(tmp_path) not in serialized
    assert "midi_path" not in serialized
    assert "audio_path" not in serialized
    assert "octave-src-" not in serialized


def test_current_like_three_song_view_with_no_test_split_is_not_admitted(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path,
        [
            _record(tmp_path, source_id, include_talkies=True)
            for source_id in _source_ids_by_split({"train": 1, "val": 2, "test": 0})
        ],
    )

    report = resolve_vocal_lead_catalog_admission(
        catalog_root=tmp_path, task_view_paths=_task_views(tmp_path)
    )

    assert report["status"] == "not_admitted"
    assert report["admission"]["passed"] is False
    assert report["data_coverage"]["source_counts"]["test"] == 0
    assert report["data_gate_outcomes"]["source_counts"]["test"] == {
        "observed": 0,
        "minimum": 10,
        "passed": False,
    }


def test_missing_required_label_is_a_nonadmitted_fail_closed_result(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path,
        [
            _record(tmp_path, source_id, include_talkies=False)
            for source_id in _source_ids_by_split({"train": 40, "val": 10, "test": 10})
        ],
    )

    report = resolve_vocal_lead_catalog_admission(
        catalog_root=tmp_path, task_view_paths=_task_views(tmp_path)
    )

    assert report["status"] == "not_admitted"
    assert report["label_availability"]["passed"] is False
    assert report["label_availability"]["missing"] == {
        "train": ["talky_spans"],
        "val": ["talky_spans"],
        "test": ["talky_spans"],
    }


def test_rejects_a_tampered_task_view_before_publishing_admission_evidence(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path,
        [
            _record(tmp_path, source_id, include_talkies=True)
            for source_id in _source_ids_by_split({"train": 1, "val": 1, "test": 1})
        ],
    )
    views = _task_views(tmp_path)
    raw = json.loads(views["vocals_talky_activity"].read_text(encoding="utf-8"))
    raw["songs"][0]["label_tracks"] = ["PART HARM1"]
    views["vocals_talky_activity"].write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(VocalLeadCatalogAdmissionError, match="cannot be revalidated"):
        resolve_vocal_lead_catalog_admission(catalog_root=tmp_path, task_view_paths=views)


def test_rejects_cross_component_source_partition_overlap_or_mismatch(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path,
        [
            _record(tmp_path, source_id, include_talkies=True)
            for source_id in _source_ids_by_split({"train": 1, "val": 1, "test": 1})
        ],
    )
    views = _task_views(tmp_path)
    raw = json.loads(views["vocals_phrase_boundaries"].read_text(encoding="utf-8"))
    raw["task"]["split_seed"] = "different-approved-partition/v1"
    from src.catalog_task_manifest import deterministic_split

    # Choose a valid alternative seed that actually changes at least one
    # selected source's split; an accidental identical three-song partition
    # is not a meaningful cross-component mismatch test.
    for suffix in range(10_000):
        candidate = f"different-approved-partition/{suffix}"
        changed = [
            deterministic_split(song["source_id"], seed=candidate) != song["split"]
            for song in raw["songs"]
        ]
        if any(changed):
            raw["task"]["split_seed"] = candidate
            break
    else:  # pragma: no cover - deterministic hash search guard.
        raise AssertionError("could not create a different valid source split")
    for song in raw["songs"]:
        song["split"] = deterministic_split(song["source_id"], seed=raw["task"]["split_seed"])
    views["vocals_phrase_boundaries"].write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(VocalLeadCatalogAdmissionError, match="source partition differs"):
        resolve_vocal_lead_catalog_admission(catalog_root=tmp_path, task_view_paths=views)


def test_worker_exposes_only_the_path_free_admission_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.worker import main

    _write_catalog(
        tmp_path,
        [
            _record(tmp_path, source_id, include_talkies=True)
            for source_id in _source_ids_by_split({"train": 1, "val": 1, "test": 1})
        ],
    )
    views = _task_views(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "strum-worker",
            "vocal",
            "lead-admission",
            "--catalog-root",
            str(tmp_path),
            "--activity-task-view",
            str(views["vocals_activity"]),
            "--phrase-task-view",
            str(views["vocals_phrase_boundaries"]),
            "--lyric-task-view",
            str(views["vocals_lyric_alignment"]),
            "--talky-task-view",
            str(views["vocals_talky_activity"]),
            "--json",
        ],
    )

    assert main() == 0

    rendered = capsys.readouterr().out
    assert json.loads(rendered)["status"] == "not_admitted"
    assert str(tmp_path) not in rendered
