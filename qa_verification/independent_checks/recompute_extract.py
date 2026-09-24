"""Stage 0 — independent extract completeness (architecture §2).

* multi-part archive presence (Zenodo `.7z.001` …)
* archive entry list vs unpacked tree (file count)
* pipeline must fail loudly on a corrupt / truncated archive
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .._lib import QAPaths, Report, load_json

ROOT = Path(__file__).resolve().parents[2]


def _find_7z() -> Optional[str]:
    for name in ("7z", "7za", "7zr"):
        p = shutil.which(name)
        if p:
            return p
    return None


def list_multipart_parts(archive: Path) -> List[Path]:
    """Return sorted `.7z.00N` parts for a multi-part set, or []."""
    archive = Path(archive)
    # file.7z.001 / file.001.7z styles
    parent = archive.parent
    stem = archive.name
    parts = sorted(parent.glob(stem + ".*"))
    parts = [p for p in parts if p.is_file() and p.suffix[1:].isdigit()]
    if parts:
        return parts
    # Maybe given the first part already
    if archive.is_file() and archive.suffix[1:].isdigit():
        base = archive.name.rsplit(".", 1)[0]
        return sorted(p for p in parent.glob(base + ".*")
                      if p.is_file() and p.suffix[1:].isdigit())
    return []


def archive_entries(archive: Path) -> Optional[List[str]]:
    """Independently list archive member paths via py7zr (or 7z l)."""
    try:
        import py7zr
        with py7zr.SevenZipFile(str(archive), mode="r") as z:
            return list(z.getnames())
    except Exception:
        pass
    exe = _find_7z()
    if exe:
        try:
            out = subprocess.check_output(
                [exe, "l", "-slt", str(archive)],
                stderr=subprocess.DEVNULL, text=True,
            )
            names = []
            for line in out.splitlines():
                if line.startswith("Path = ") and not line.endswith(archive.name):
                    names.append(line.split("Path = ", 1)[1])
            return names or None
        except Exception:
            return None
    return None


def count_files(root: Path) -> int:
    n = 0
    for _dp, _dn, files in root.walk() if hasattr(root, "walk") else _os_walk(root):
        n += len(files)
    return n


def _os_walk(root: Path):
    import os
    return os.walk(root)


def run(paths: QAPaths, *, sample: int = 30, seed: int = 42) -> Report:
    rep = Report("stage0_extract")

    stage = paths.stage_dir
    if not stage.is_dir():
        rep.hard(False, f"stage_dir missing: {stage}")
        return rep

    # --- multi-part completeness (Zenodo) ---------------------------------
    archive = paths.source_archive
    if archive is None:
        # discover a nearby raw archive under data/raw or data/
        candidates = list((ROOT / "data").rglob("*.7z"))
        candidates += list((ROOT / "data").rglob("*.7z.001"))
        archive = candidates[0] if candidates else None

    if archive is not None and Path(archive).exists():
        archive = Path(archive)
        parts = list_multipart_parts(archive)
        if len(parts) >= 1 and archive.name.endswith((".001", ".002")) or len(parts) > 1:
            # verify sequential part numbers
            nums = []
            for p in parts:
                try:
                    nums.append(int(p.suffix.lstrip(".")))
                except ValueError:
                    pass
            nums.sort()
            expected = list(range(1, (max(nums) if nums else 0) + 1))
            missing_parts = sorted(set(expected) - set(nums))
            rep.hard(
                not missing_parts,
                "multi-part archive parts contiguous" if not missing_parts
                else f"multi-part missing parts: {missing_parts}",
                parts=[p.name for p in parts],
                missing=missing_parts,
            )
            if missing_parts:
                # must NOT silently proceed — stage_dir should be incomplete or empty-ish
                rep.info("missing parts present — pipeline must have errored before a full tree")

        entries = archive_entries(archive) if archive.is_file() or archive.name.endswith(".001") else None
        if entries is not None:
            # count real files in stage (skip directory-only entries)
            file_entries = [e for e in entries if not str(e).endswith("/")]
            n_stage = sum(1 for p in stage.rglob("*") if p.is_file())
            # multi-part / nested layouts: allow stage to contain ≥ file entries
            # or at least be non-empty when archive had files
            if file_entries:
                ok = n_stage >= 1 and (
                    n_stage >= len(file_entries) * 0.5  # tolerate empty-dir markers
                    or n_stage >= len(file_entries)
                )
                # stricter: every archive file basename should appear somewhere
                stage_names = {p.name for p in stage.rglob("*") if p.is_file()}
                missing = []
                for e in file_entries[:500]:
                    base = Path(str(e)).name
                    if base and base not in stage_names:
                        # multi-part solid archives may rename — only flag if many missing
                        missing.append(base)
                frac_missing = len(missing) / max(1, min(len(file_entries), 500))
                rep.hard(
                    frac_missing < 0.05,
                    "extracted tree covers archive entries"
                    if frac_missing < 0.05
                    else f"extract missing many archive entries ({len(missing)} sampled)",
                    n_archive_files=len(file_entries),
                    n_stage_files=n_stage,
                    missing_sample=missing[:10],
                )
    else:
        rep.info("no source archive located — skipping archive↔tree completeness")

    # --- corrupt archive must fail (graceful) -----------------------------
    # Create a truncated garbage .7z and confirm extract exits non-zero.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "corrupt.7z"
        bad.write_bytes(b"7z\xbc\xaf\x27\x1c" + b"\x00" * 64)  # bad header/body
        out_dir = Path(td) / "out"
        script = ROOT / "sar_dataset_pipeline.py"
        try:
            proc = subprocess.run(
                [sys.executable, str(script), "extract",
                 "--archive", str(bad), "--stage_dir", str(out_dir)],
                capture_output=True, text=True, timeout=60,
                cwd=str(ROOT),
            )
            failed = proc.returncode != 0
            rep.hard(
                failed,
                "corrupt archive extract fails (non-zero exit)"
                if failed else "corrupt archive extract returned 0 — silent failure risk",
                returncode=proc.returncode,
                stderr_tail=(proc.stderr or "")[-400:],
            )
        except Exception as exc:
            rep.soft(False, f"corrupt-archive probe raised: {exc}")

    # --- unpacked tree non-empty -----------------------------------------
    n_files = sum(1 for p in stage.rglob("*") if p.is_file())
    rep.hard(n_files > 0, "stage_dir contains files" if n_files else "stage_dir is empty",
             n_files=n_files)
    return rep
