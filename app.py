"""Flask API and Three.js viewer with per-GLB, rebuildable spatial indexes."""

from __future__ import annotations

import os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, url_for
import numpy as np
from sentence_transformers import SentenceTransformer

from index_service import IndexManager
from scene_metadata import scene_fingerprint

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SCENES_DIR = os.path.join(APP_DIR, "scenes")
app = Flask(__name__)
MODEL_NAME = "google/embeddinggemma-2"
model = SentenceTransformer(MODEL_NAME)
manager = IndexManager(APP_DIR, model)

def scene_paths():
    return {path.parent.name: path for path in sorted(Path(SCENES_DIR).glob("*/*.glb"))}


def scene_catalog():
    scenes = []
    for name, path in scene_paths().items():
        snapshot = manager.get(scene_fingerprint(path))
        scenes.append({"name": name,
            "glb_url": url_for("glb", filename=path.relative_to(SCENES_DIR).as_posix()),
            "indexed": snapshot is not None, "up_axis": "z",
            "index": snapshot.info() if snapshot else None})
    return scenes


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "viewer.html", max_age=0)


@app.route("/glb/<path:filename>")
def glb(filename):
    return send_from_directory(SCENES_DIR, filename, mimetype="model/gltf-binary")


@app.route("/scenes")
def list_scenes():
    return jsonify(scene_catalog())


@app.post("/reindex")
def reindex():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify(error="Expected a JSON object with a scene name."), 400
    name = data.get("scene")
    path = scene_paths().get(name) if isinstance(name, str) else None
    if path is None:
        return jsonify(error="Choose an existing GLB scene."), 400
    try:
        voxel_size = float(data.get("voxel_size", 0.5))
    except (TypeError, ValueError):
        return jsonify(error="Patch size must be 0.25, 0.5, or 1 meter."), 400
    if voxel_size not in (0.25, 0.5, 1.0):
        return jsonify(error="Patch size must be 0.25, 0.5, or 1 meter."), 400
    try:
        job = manager.start(path, name, voxel_size)
    except RuntimeError as error:
        return jsonify(error=str(error), job=manager.status()), 409
    return jsonify(job=job), 202


@app.get("/index/status")
def index_status():
    name = request.args.get("scene")
    path = scene_paths().get(name)
    if name and path is None:
        return jsonify(error="Unknown scene."), 404
    snapshot = manager.get(scene_fingerprint(path)) if path else None
    return jsonify(job=manager.status(), scene=name,
        index=snapshot.info() if snapshot else None)


@app.get('/memory/<scene_name>/<generation>/<filename>')
def memory_image(scene_name, generation, filename):
    path = scene_paths().get(scene_name)
    snapshot = manager.get(scene_fingerprint(path)) if path else None
    if not snapshot or not snapshot.visual or snapshot.metadata.get('generation') != generation:
        return jsonify(error='This observation is no longer in the active memory.'), 404
    return send_from_directory(snapshot.visual.directory/'observations', filename)


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify(error="missing q"), 400
    try:
        k = min(100, max(1, int(request.args.get("k", "8"))))
    except ValueError:
        k = 8
    name = request.args.get("scene")
    paths = scene_paths()
    if not name:
        name = next((key for key, path in paths.items() if manager.get(scene_fingerprint(path))), None)
    if name not in paths:
        return jsonify(error="Unknown scene."), 404
    snapshot = manager.get(scene_fingerprint(paths[name]))
    if snapshot is None:
        return jsonify(error=f"No memory for {name}. Click Rebuild memory first."), 409
    response = {"query": query, "scene": name, "count": 0, "results": [],
        "generation": snapshot.metadata.get("generation"),
        "retrieval_mode": snapshot.info()['retrieval_mode'], "visual_semantics": bool(snapshot.visual)}
    # A single query supports both surface descriptions and things seen in the
    # walkthrough. Only literal surface queries use geometry ranking; everything
    # else searches actual images, with no object-name allowlist or denylist.
    surface_words = {'a','the','an','surface','surfaces','wall','walls','floor','floors',
        'ceiling','ceilings','horizontal','vertical','up','down','facing'}
    literal_surface = set(query.lower().split()) <= surface_words
    if snapshot.visual and not literal_surface:
        results = snapshot.visual.search(query, manager.visual_encoder, k)
        units_by_id = {unit['unit_id']:unit for unit in snapshot.units}
        for result in results:
            result['patches'] = [units_by_id[key] for key in result['patch_ids'] if key in units_by_id]
            result['image_url'] = url_for('memory_image',scene_name=name,
                generation=snapshot.metadata['generation'], filename=result['image'])
            result['position'] = result['camera']['position']
            result['target_position'] = result['center']
            result['view_angle'] = {key:result['camera'][key] for key in ('yaw','pitch')}
        response.update(supported=True, count=len(results), results=results)
        return jsonify(response)
    with manager.inference_lock:
        embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float64)
    embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
    scores = snapshot.embeddings @ embedding
    surface = request.args.get("surface")
    candidates = [i for i, unit in enumerate(snapshot.units) if not surface or unit["surface_type"] == surface]
    if snapshot.visual:
        candidates = [i for i in candidates if snapshot.units[i]['unit_id'] in snapshot.surface_views]
    # Equivalent descriptions have identical embeddings. Prefer a substantial
    # surface patch over a tiny scan fragment when their text scores tie.
    order = sorted(candidates, key=lambda i: (-float(scores[i]), -snapshot.units[i]["area"]))[:k]
    results = [{**snapshot.units[i], "similarity": float(scores[i])} for i in order]
    for result in results:
        saved = (snapshot.surface_views or {}).get(result['unit_id'])
        if saved:
            result['camera'] = saved
            result['position'] = result['camera']['position']
            result['view_angle'] = {key:result['camera'][key] for key in ('yaw','pitch')}
            result['target_position'] = saved['target']
    if not snapshot.visual:
        response['message'] = 'Reindex this scene to add eye-level visual memories. The current index contains surface descriptions.'
    response.update(supported=True, count=len(results), results=results)
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
