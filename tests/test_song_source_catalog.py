import hashlib
import json
from pathlib import Path

import pytest

from src.song_source_catalog import CatalogValidationError, load_catalog, select_training_sources

_NO_CURATION = object()


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


def _record(root: Path, source_id: str, training_use: str = "allowed") -> dict[str, object]:
    return {
        "source_id": source_id,
        "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
        "rights": {
            "training_use": training_use,
            "provenance": "Reviewed local collection",
            "license": "test-only",
        },
        "metadata": {"name": "Safe Song", "artist": "Safe Artist"},
        "chart": {
            "notes_midi": _asset(root, b"MThd test", "notes.mid"),
            "instruments": {
                "guitar": {
                    "status": "present",
                    "difficulties": ["hard", "expert"],
                    "track_names": ["PART GUITAR"],
                },
                "vocals": {"status": "absent", "difficulties": [], "track_names": []},
            },
        },
        "audio": {"guitar": _asset(root, b"local song audio", "guitar.ogg")},
    }


def _catalog(
    root: Path, records: list[dict[str, object]], *, curation: object = _NO_CURATION
) -> None:
    manifest: dict[str, object] = {
        "schema_version": 1,
        "format": "octave-song-source-catalog/v1",
        "catalog_id": "local-test",
        "records": "records.jsonl",
    }
    if curation is not _NO_CURATION:
        manifest["curation"] = curation
    (root / "records.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    (root / "catalog.json").write_text(json.dumps(manifest))


def test_loads_allowed_instrument_sources_without_exposing_import_locations(tmp_path: Path) -> None:
    allowed = _record(tmp_path, "octave-src-12345678")
    review = _record(tmp_path, "octave-src-abcdefgh", training_use="review_required")
    _catalog(tmp_path, [allowed, review])

    catalog = load_catalog(tmp_path)
    sources = select_training_sources(
        catalog, "guitar", required_difficulties=["expert", "hard"], audio_role="guitar"
    )

    assert [source.source_id for source in sources] == ["octave-src-12345678"]
    assert sources[0].audio is not None
    assert not hasattr(sources[0], "provenance")


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "/home/ash/private",
        "smb://minipc.local/songs",
        "C:\\secret\\song",
        "source=/home/ash/private",
        "ref=https://example.invalid/song",
        "ref=s3://bucket/private/source",
    ],
)
def test_rejects_path_or_url_text_before_training_views(tmp_path: Path, unsafe_value: str) -> None:
    record = _record(tmp_path, "octave-src-12345678")
    record["rights"]["provenance"] = unsafe_value  # type: ignore[index]
    _catalog(tmp_path, [record])

    with pytest.raises(CatalogValidationError, match="unsafe text"):
        load_catalog(tmp_path)


def test_rejects_asset_under_a_different_hash_directory(tmp_path: Path) -> None:
    record = _record(tmp_path, "octave-src-12345678")
    midi = record["chart"]["notes_midi"]  # type: ignore[index]
    midi["relative_path"] = "assets/sha256/" + "0" * 64 + "/notes.mid"  # type: ignore[index]
    _catalog(tmp_path, [record])

    with pytest.raises(CatalogValidationError, match="content address"):
        load_catalog(tmp_path)


def test_rejects_unstructured_warnings_and_unknown_path_fields(tmp_path: Path) -> None:
    record = _record(tmp_path, "octave-src-12345678")
    record["import"]["warnings"] = [{"message": "/private/package.sng"}]  # type: ignore[index]
    record["original_path"] = "/private/package.sng"
    _catalog(tmp_path, [record])

    with pytest.raises(CatalogValidationError):
        load_catalog(tmp_path)


def test_returns_training_sources_in_source_id_order(tmp_path: Path) -> None:
    later = _record(tmp_path, "octave-src-zzzzzzzz")
    earlier = _record(tmp_path, "octave-src-aaaaaaaa")
    _catalog(tmp_path, [later, earlier])

    catalog = load_catalog(tmp_path)

    assert [source.source_id for source in select_training_sources(catalog, "guitar")] == [
        "octave-src-aaaaaaaa",
        "octave-src-zzzzzzzz",
    ]


def test_allows_normal_metadata_with_a_non_path_slash(tmp_path: Path) -> None:
    record = _record(tmp_path, "octave-src-12345678")
    record["metadata"]["artist"] = "AC/DC"  # type: ignore[index]
    _catalog(tmp_path, [record])

    assert load_catalog(tmp_path).records[0].source_id == "octave-src-12345678"


def test_accepts_safe_catalog_curation_without_exposing_it_to_training(tmp_path: Path) -> None:
    _catalog(
        tmp_path,
        [_record(tmp_path, "octave-src-12345678")],
        curation={
            "provenance": "Curated in OCTAVE",
            "license": "Permission recorded by catalog owner",
        },
    )

    catalog = load_catalog(tmp_path)

    assert catalog.records[0].training_use == "allowed"
    assert not hasattr(catalog, "curation")
    assert not hasattr(catalog.records[0], "curation")


@pytest.mark.parametrize(
    "curation",
    [
        {"provenance": "Curated in OCTAVE"},
        {
            "provenance": "Curated in OCTAVE",
            "license": "Permission recorded by catalog owner",
            "source_path": "/tmp/private",
        },
        {"provenance": "/tmp/private", "license": "Permission recorded by catalog owner"},
        {"provenance": "private/source.sng", "license": "Permission recorded by catalog owner"},
        {"provenance": "Curated in OCTAVE", "license": "x" * 513},
        ["Curated in OCTAVE", "Permission recorded by catalog owner"],
        None,
    ],
)
def test_rejects_unsafe_or_noncanonical_catalog_curation(tmp_path: Path, curation: object) -> None:
    _catalog(tmp_path, [_record(tmp_path, "octave-src-12345678")], curation=curation)

    with pytest.raises(CatalogValidationError, match="curation"):
        load_catalog(tmp_path)
