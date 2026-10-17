"""Track which GLB a single-scene search index belongs to, even after renames."""

from functools import lru_cache
from hashlib import file_digest
import json
from pathlib import Path


@lru_cache(maxsize=64)
def _fingerprint(path: str, size: int, mtime_ns: int) -> str:
    with open(path, "rb") as source:
        return file_digest(source, "sha256").hexdigest()


def scene_fingerprint(path) -> str:
    source = Path(path).resolve()
    stat = source.stat()
    return _fingerprint(str(source), stat.st_size, stat.st_mtime_ns)


def save_index_metadata(scene_path, path_prefix="index_data", **details):
    metadata = {
        "scene_path": str(scene_path) if scene_path is not None else None,
        "scene_sha256": scene_fingerprint(scene_path) if scene_path is not None else None,
        "up_axis": "z",
        **details,
    }
    Path(f"{path_prefix}.meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_index_metadata(path_prefix="index_data"):
    path = Path(f"{path_prefix}.meta.json")
    return json.loads(path.read_text()) if path.exists() else {}
