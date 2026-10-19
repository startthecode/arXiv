import os
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    """Write via temp file + rename, so a crash never leaves a half-written file behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
