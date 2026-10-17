"""Embed unit descriptions with google/embeddinggemma-2 and save for indexing."""

from __future__ import annotations

import json
import sys
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from voxelize import voxelize
from parse_glb import load_scene
from scene_metadata import save_index_metadata

MODEL_NAME = "google/embeddinggemma-2"


def embed_units(units, model_name: str = MODEL_NAME, batch_size: int = 32,
                model=None, on_progress=None):
    """Return (embeddings, units) with L2-normalized embeddings (cosine-ready)."""
    if model is None:
        model = SentenceTransformer(model_name)
    descriptions = [u.description for u in units]
    # Coordinates belong in spatial fields, not semantic text. All patches
    # with the same surface description should receive the same embedding.
    unique, inverse = np.unique(descriptions, return_inverse=True)
    batches = []
    for offset in range(0, len(unique), batch_size):
        batches.append(model.encode(unique[offset:offset+batch_size].tolist(),
            convert_to_numpy=True, normalize_embeddings=True))
        if on_progress:
            on_progress(min(offset+batch_size, len(unique))/len(unique))
    embeddings = np.concatenate(batches, axis=0)[inverse]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    # Re-normalize to be safe (sentence-transformers already normalizes)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    return embeddings, units


def save(embeddings: np.ndarray, units, path_prefix: str = "index_data", scene_path=None,
         metadata=None):
    np.save(f"{path_prefix}.npy", embeddings)
    serial = []
    for u in units:
        serial.append({
            "unit_id": u.unit_id,
            "voxel": list(u.voxel),
            "center": [float(v) for v in u.center],
            "bbox_min": [float(v) for v in u.bbox_min],
            "bbox_max": [float(v) for v in u.bbox_max],
            "cell_min": [float(v) for v in u.cell_min],
            "cell_max": [float(v) for v in u.cell_max],
            "orientation": u.orientation,
            "surface_type": u.surface_type,
            "facing": u.facing,
            "normal": [float(v) for v in u.normal],
            "triangle_count": u.triangle_count,
            "area": float(u.area),
            "description": u.description,
            "node_names": u.node_names,
        })
    with open(f"{path_prefix}.json", "w") as f:
        json.dump(serial, f, indent=2)
    save_index_metadata(scene_path, path_prefix, **(metadata or {}))
    print(f"saved {len(units)} units -> {path_prefix}.npy ({embeddings.shape}) + {path_prefix}.json")


def main():
    glb = sys.argv[1] if len(sys.argv) > 1 else "scenes/sample-room/sample-room.glb"
    voxel_size = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    scene = load_scene(glb)
    units = voxelize(scene, voxel_size=voxel_size)
    embeddings, units = embed_units(units)
    save(embeddings, units, scene_path=glb, metadata={
        "voxel_size": voxel_size, "bounds_kind": "clipped_surface_aabb",
        "geometry_version": 2, "retrieval_mode": "geometry_text", "visual_semantics": False,
        "glb_bounds_min": scene.bounds_min.tolist(), "glb_bounds_max": scene.bounds_max.tolist(),
        "source_triangle_count": len(scene.triangles),
    })


if __name__ == "__main__":
    main()
