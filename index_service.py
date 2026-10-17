"""Background full-file indexing with per-GLB snapshots and atomic activation."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

import duckdb
import numpy as np

from build_index import DB_PATH, TABLE, build_index
from embed import embed_units, save
from parse_glb import load_scene
from scene_metadata import scene_fingerprint
from visual_memory import VisualEncoder, VisualMemory, build_visual_memory
from voxelize import voxelize


@dataclass(frozen=True)
class Snapshot:
    units: list
    embeddings: np.ndarray
    metadata: dict
    visual: VisualMemory | None = None
    surface_views: dict | None = None

    def info(self):
        return {"retrieval_mode": "geometry_text", "visual_semantics": False,
                **self.metadata, "unit_count": len(self.units)}


def read_snapshot(path):
    with duckdb.connect(str(path), read_only=True) as connection:
        has_metadata = connection.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [f"{TABLE}_metadata"]).fetchone()[0]
        metadata = json.loads(connection.execute(f"SELECT metadata FROM {TABLE}_metadata").fetchone()[0]) if has_metadata else {}
        cursor = connection.execute(f"SELECT * FROM {TABLE} ORDER BY unit_id")
        columns = [column[0] for column in cursor.description]
        records = [dict(zip(columns, row)) for row in cursor.fetchall()]
    units = []
    for row in records:
        unit = {key: row[key] for key in ("unit_id", "surface_type", "facing", "description", "area")}
        for field in ("center", "bbox_min", "bbox_max", "normal"):
            unit[field] = [row[f"{field}_{axis}"] for axis in "xyz"]
        for field, fallback in (("cell_min", "bbox_min"), ("cell_max", "bbox_max")):
            unit[field] = [row.get(f"{field}_{axis}", row[f"{fallback}_{axis}"]) for axis in "xyz"]
        unit["orientation"] = row.get("orientation", -1)
        units.append(unit)
    embeddings = np.asarray([row["embedding"] for row in records], dtype=np.float64)
    if len(embeddings):
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    directory = Path(path).parent
    visual = VisualMemory.read(directory) if metadata.get('memory_version') else None
    surface_views = json.loads((directory/'observations.json').read_text()).get('surface_views',{}) if visual else {}
    return Snapshot(units, embeddings, metadata, visual, surface_views)


class IndexManager:
    def __init__(self, root, model, visual_encoder=None):
        self.root = Path(root)
        self.directory = self.root / ".indexes"
        self.model = model
        self.visual_encoder = visual_encoder or VisualEncoder(root)
        self.inference_lock = threading.Lock()
        self.lock = threading.RLock()
        self.snapshots = {}
        self.job = None
        legacy = self.root / DB_PATH
        if legacy.exists():
            snapshot = read_snapshot(legacy)
            fingerprint = snapshot.metadata.get("scene_sha256")
            if fingerprint:
                self.snapshots[fingerprint] = snapshot

    def get(self, fingerprint):
        with self.lock:
            pointer = self.directory / fingerprint / "current.json"
            if pointer.exists():
                version = json.loads(pointer.read_text())["generation"]
                existing = self.snapshots.get(fingerprint)
                if existing is None or existing.metadata.get("generation") != version:
                    self.snapshots[fingerprint] = read_snapshot(pointer.parent / version / "memory.duckdb")
            return self.snapshots.get(fingerprint)

    def status(self):
        with self.lock:
            return dict(self.job) if self.job else None

    def start(self, path, scene_name, voxel_size):
        fingerprint = scene_fingerprint(path)
        with self.lock:
            if self.job and self.job["status"] == "running":
                raise RuntimeError("Another full-file index is already running.")
            self.job = {"id": uuid.uuid4().hex, "scene": scene_name,
                "status": "running", "phase": "starting", "progress": 0,
                "message": "Preparing the entire GLB", "voxel_size": voxel_size}
            job = dict(self.job)
        threading.Thread(target=self._run, args=(Path(path), fingerprint, job), daemon=True).start()
        return job

    def _progress(self, phase, progress, message):
        with self.lock:
            self.job.update(phase=phase, progress=round(progress, 1), message=message)

    def _run(self, path, fingerprint, job):
        try:
            self._progress("parsing", 3, "Reading every mesh and applying GLB world transforms")
            scene = load_scene(str(path))
            if not scene.triangles:
                raise ValueError("The GLB has no triangle geometry to index.")
            count = len(scene.triangles)
            self._progress("geometry", 12, f"Clipping {count:,} triangles into surface patches")
            units = voxelize(scene, voxel_size=job["voxel_size"], min_area=0,
                on_progress=lambda progress:self._progress("geometry", 12+progress*43,
                    f"Processing all {count:,} triangles · {progress:.0%}"))
            if not units:
                raise ValueError("No non-degenerate surface patches were found.")
            indexed_area = sum(unit.area for unit in units)
            if not np.isclose(indexed_area, scene.surface_area, rtol=1e-7, atol=1e-8):
                raise ValueError("Surface coverage validation failed; the previous index remains active.")
            self._progress("embedding", 57, f"Embedding descriptions for {len(units):,} surface patches")
            with self.inference_lock:
                embeddings, units = embed_units(units, model=self.model,
                    on_progress=lambda progress:self._progress("embedding", 57+progress*3,
                        "Embedding surface descriptions"))
            metadata = {"geometry_version": 2, "bounds_kind": "clipped_surface_aabb",
                "voxel_size": job["voxel_size"], "retrieval_mode": "geometry_text",
                "visual_semantics": False, "generation": job["id"],
                "built_at": datetime.now(timezone.utc).isoformat(),
                "source_triangle_count": count,
                "source_surface_area": scene.surface_area, "indexed_surface_area": indexed_area,
                "glb_bounds_min": scene.bounds_min.tolist(), "glb_bounds_max": scene.bounds_max.tolist()}
            version = self.directory / fingerprint / job["id"]
            version.mkdir(parents=True)
            self._progress('walkthrough',60,'Planning eye-level views and loading the visual memory model')
            visual_metadata = build_visual_memory(scene,units,version,self.visual_encoder,
                on_progress=lambda progress,message:self._progress('walkthrough',60+progress*30,message))
            metadata.update(visual_metadata)
            prefix = str(version / "data")
            self._progress("saving", 92, "Saving geometry, visual memories, depth registration and camera poses")
            save(embeddings, units, prefix, scene_path=path, metadata=metadata)
            build_index(prefix, str(version / "memory.duckdb"))
            snapshot = read_snapshot(version / "memory.duckdb")
            if scene_fingerprint(path) != fingerprint:
                raise ValueError("The GLB changed during indexing. Run the rebuild again.")
            # Queries continue using the old complete snapshot until this point.
            pointer = version.parent / "current.json"
            pending = version.parent / f"{job['id']}.pending.json"
            pending.write_text(json.dumps({"generation": job["id"]}))
            with self.lock:
                pending.replace(pointer)
                self.snapshots[fingerprint] = snapshot
                self.job.update(status="complete", phase="complete", progress=100,
                    unit_count=len(units), source_triangle_count=count,
                    observation_count=metadata['observation_count'],
                    message=f"Remembered {metadata['observation_count']:,} observations from {metadata['walkthrough_view_count']:,} eye-level views; {len(units):,} surface patches")
        except Exception as error:
            with self.lock:
                self.job.update(status="failed", phase="failed", message=str(error))
