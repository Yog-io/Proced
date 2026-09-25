"""Stage 0 / Stage 6 — .7z decompression and archival.

Prefers the system ``7z`` CLI (multi-threaded, fast). Falls back to py7zr when
the CLI is unavailable (the original archi script crashed outright without
p7zip installed).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .progress import bar as progress_bar

log = logging.getLogger("proced.archive")


def _find_7z() -> Optional[str]:
    for name in ("7z", "7za", "7zr"):
        p = shutil.which(name)
        if p:
            return p
    return None


def unpack_7z(archive_path: Path, extract_dir: Path) -> Path:
    """Stage 0: multi-core decompression into ``extract_dir`` (py7zr fallback)."""
    archive_path, extract_dir = Path(archive_path), Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    exe = _find_7z()
    if exe:
        log.info("Decompressing %s into %s (7z CLI, multithreaded)...", archive_path, extract_dir)
        cmd = [exe, "x", f"-o{extract_dir}", str(archive_path), "-y", "-mmt=on"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return extract_dir

    import py7zr
    log.info("Decompressing %s into %s (py7zr fallback)...", archive_path, extract_dir)
    with py7zr.SevenZipFile(str(archive_path), mode="r") as z:
        z.extractall(path=str(extract_dir))
    return extract_dir


def pack_7z(source_dir: Path, output_archive: Path) -> Path:
    """Stage 6: pack the processed tree into a compressed .7z archive."""
    source_dir, output_archive = Path(source_dir), Path(output_archive)
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    exe = _find_7z()
    if exe:
        log.info("Compressing %s into %s (7z CLI)...", source_dir, output_archive)
        # 7z expands the wildcard itself when no shell is involved.
        cmd = [exe, "a", "-t7z", "-m0=lzma2", "-mx=7", "-mmt=on",
               str(output_archive), f"{source_dir.name}/*"]
        subprocess.run(cmd, check=True, cwd=str(source_dir.parent),
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return output_archive

    import py7zr
    log.info("Compressing %s into %s (py7zr fallback)...", source_dir, output_archive)
    entries = sorted(source_dir.rglob("*"))
    pb = progress_bar(len(entries), "archive pack", unit="entry")
    try:
        with py7zr.SevenZipFile(str(output_archive), mode="w") as z:
            for p in entries:
                # Write directories too — empty dirs must survive the round-trip
                # (directory-equivalence check depends on the full tree).
                z.write(str(p), str(p.relative_to(source_dir)))
                pb.update()
    finally:
        pb.close()
    return output_archive
