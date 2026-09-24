"""Archive: py7zr/7z round-trip."""

from pathlib import Path

import pytest

from proced.archive import pack_7z, unpack_7z


def test_pack_unpack_roundtrip(tmp_path):
    src = tmp_path / "data_out"
    (src / "region" / "sub").mkdir(parents=True)
    (src / "region" / "sub" / "a.png").write_bytes(b"\x89PNG fake")
    (src / "region" / "sub" / "GEODATA.csv").write_text("crop_id\nx")
    (src / "metadata.csv").write_text("crop_id\nx")

    arc = tmp_path / "out.7z"
    pack_7z(src, arc)
    assert arc.is_file() and arc.stat().st_size > 0

    dest = tmp_path / "restored"
    unpack_7z(arc, dest)
    # Pack archives dest contents at root → restored as data_out/... or directly
    found = list(dest.rglob("a.png"))
    assert found, f"round-trip lost files; tree={list(dest.rglob('*'))}"
    csvs = list(dest.rglob("GEODATA.csv"))
    assert csvs


def test_unpack_missing_archive_raises(tmp_path):
    with pytest.raises(Exception):
        unpack_7z(tmp_path / "nope.7z", tmp_path / "x")
