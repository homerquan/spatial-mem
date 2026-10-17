# 3D Spatial Memory — LiDAR Room Search

A local spatial memory for a textured GLB. The indexer takes a virtual
walkthrough at eye level, remembers what it sees, and registers every observation
to the scan. One search finds wall art, physical objects, and geometric surfaces.
Results include camera position, yaw/pitch, a recorded image, and visible 3D
patches. Clicking a result restores its viewpoint.

Visual search uses [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224)
image/text embeddings. Surface descriptions use EmbeddingGemma-2 and DuckDB.
Inference and scan processing run locally; the first build downloads model
weights, without uploading scan images to a service.

## Pipeline

```
Textured GLB → world-space triangles, UVs and textures
                 ├─ voxelize.py → fitted surface patches → text embeddings
                 └─ memory_render.py → eye-level RGB + depth + triangle IDs
                                      → full-view and overlapping crop memories
                                      → SigLIP 2 image embeddings
                                      → camera pose + visible patch IDs
Free-text query → image/text retrieval → position + view angle + observed region
                                      → Three.js viewpoint and surface heatmap
```

## Files

| File | Purpose |
|------|---------|
| `parse_glb.py` | Load GLB → world-space triangles, normals, semantic labels |
| `voxelize.py` | Clip every triangle into surface patches and fit their bounds |
| `geometry.py` | World-space triangle clipping used by the indexer |
| `index_service.py` | Background rebuild jobs, progress, per-GLB snapshots |
| `memory_render.py` | Textured first-person RGB/depth rendering and walkthrough planning |
| `visual_memory.py` | Local image embeddings, registered observations and view retrieval |
| `embed.py` | Embed descriptions with `google/embeddinggemma-2` |
| `build_index.py` | Build DuckDB vector index (`spatial_memory.duckdb`) |
| `query.py` | CLI text lookup |
| `app.py` | Flask search API + static/GLB server |
| `viewer.html` | three.js 3D viewer with search & 3D highlights |
| `index_data.npy / .json` | Embeddings + unit records (generated) |
| `index_data.meta.json` | Source GLB fingerprint and coordinate convention |
| `spatial_memory.duckdb` | Persistent vector index (generated) |

## Quick start

Scan datasets, model weights, and generated indexes are not bundled in Git.
Place each GLB and any external textures under `scenes/<scene-name>/` before
starting the app. The GLB filename can differ from its directory name.

```bash
# 1. create venv & install deps (python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the app
python app.py
# open http://127.0.0.1:5050/
# 3. Select a scene and click Rebuild memory; then search.
```

In the viewer, select a scene and click **Rebuild memory**. This reads
every mesh in the file, applies its world transforms, clips crossing triangles
into patches, embeds surface descriptions, and walks through supported floor
locations at 1.6 m eye height with eight headings per position. Each recorded
view includes RGB, depth, source triangle IDs, visible patch IDs, and a Z-up
camera pose. Overlapping image crops improve retrieval of smaller objects.
All memories and the surface index activate together as one new snapshot.
Progress is shown in the viewer. Patch sizes are 0.25, 0.5 (default), or 1 m.
The old index remains usable until the rebuild succeeds; a failed rebuild does
not replace it. Completed indexes activate immediately, without restarting Flask.

Indexes are saved per GLB content fingerprint under `.indexes/`, so indexing
another scene preserves earlier scene indexes and renaming a file does not
break its association. The viewer uses actual GLB paths, so directory and file
names can differ. The original CLI workflow above still writes the root
`spatial_memory.duckdb`, which serves as a fallback for older installations.
Older indexes need **Rebuild memory** once to add visual observations. The
original CLI commands (`embed.py`, `build_index.py`, `query.py`) still support
the surface-description workflow; visual indexing runs through `index_service.py`.

Each completed version contains `memory.duckdb`, `visual.npy`,
`observations.json`, observation JPEGs, and `frames/` with full RGB images,
compressed depth/triangle/patch maps and `poses.json`. Query results persist
across restarts. The SigLIP checkpoint revision is pinned and recorded in metadata.

The HM3D scans and search coordinates use Z-up. The camera, grid, highlights,
and fly-to controls use the same convention. Highlight boxes fit the actual
world-space extents of clipped surface geometry, with perpendicular surfaces
kept separate. They are not full cell cubes or detected object boxes. Heatmaps
use the same clipping boundaries and surface orientations as indexing; triangles
that cross a boundary are clipped rather than selected by centroid alone.
Labels attach to the surface and respect occlusion.

Search highlights include a 55%-opacity surface heatmap with soft falloff.
For visual results, heatmaps cover source patches visible in the retrieved
observation, rather than all geometry within an enclosing box. Visual bounds
describe that observed region; they are not segmented object outlines.
Use **Heatmap**, **Boxes**, and **Dim original** to adjust the overlay.
Warm colors indicate higher-ranked results, with cooler colors for lower ranks.

Click **Walk through** or press **V** to enter free walk mode. **WASD** moves,
**Q/E** moves down/up, **Shift** increases speed, and the arrow keys move/turn.
Click the 3D view for mouse look, or drag if pointer lock is unavailable.
**Esc** returns to orbit; **Room overview** restores the full-scene view.
Typing in the search field does not move the camera. Free walk has no wall
collision detection.

Run API, asset-path, and coordinate regression checks without loading the
embedding model:

```bash
python -m unittest test_viewer.py test_visual_memory.py
```

For heatmap geometry, load races, keyboard movement, occlusion-aware focusing
and camera-pose restoration checks, use the same
Three.js module as the viewer (or an installed `three` package):

```bash
THREE_MODULE=/path/to/three.module.js node test_viewer.mjs
```

## Why a painting query previously highlighted a different wall

The original sample-room index had 701 descriptions and zero painting,
picture, or frame labels. Its descriptions came from surface normals and
height, without examining the 41 image textures in the GLB. A query such as
"painting on the wall" therefore matched generic wall text; it did not locate
the painting. Coordinates embedded into those generic descriptions also biased
the ranking among otherwise equivalent walls. The sample GLB has no node
transforms that could mirror left and right.

The rebuild keeps coordinates in spatial fields and embeds location-free
surface descriptions. It also indexes actual textured walkthrough images.
Object queries search these images, while literal surface queries use visible
surface descriptions and their recorded viewpoints. There is no separate
surface/object mode and no blocked object-name list. RGB, depth and patch
registration are generated from the same triangles and transforms.

Previously, result-click navigation guessed a camera offset from an averaged
normal, which could place the camera behind a wall. Visual results now restore
the recorded camera pose and crop field of view. Surface results aim at a point
actually seen from their saved position; legacy results raycast candidate views
and reject intervening geometry. Orbit damping is cleared before restoring a pose.

## Retrieval coordinates

`GET /search?q=painting%20on%20the%20wall&scene=sample-room&k=5` returns:

* `position`: camera position in GLB world meters, with Z up.
* `view_angle`: yaw in degrees from +X toward +Y, and pitch toward +Z.
* `camera`: position, normalized viewing direction, target, up axis and vertical FOV.
* `target_position` / `center`: location of the observed region in the scene.
* `capture_camera` and `crop`: original view and image crop for visual memories.
* `image_url`: recorded observation; `patches`: visible registered geometry.

The viewer displays position and angles on each result and restores them on
click or Enter. The 3D viewport excludes the control sidebar when framing views.

## Notes & limitations

* Results are visual similarity matches, not detector confidence or proof an
  object exists. Regions can include surrounding context and multiple objects.
* The full geometry is indexed, but visual memory covers what the walkthrough
  can see. Hidden surfaces, closed rooms and very small details can be missed.
  Coverage counts are available in Index notes. A supported floor and adequate
  headroom are required for a visual walkthrough.
* SigLIP inference uses Apple MPS when available and otherwise CPU. Model weights
  are cached under `.models/`. Large scenes and CPU-only builds take longer.
* `array_cosine_similarity` binding via DuckDB parameters is finicky in this
  DuckDB build, so `query.py`/`app.py` fetch stored (normalized) embeddings
  and rank by dot-product in numpy — equivalent to a brute-force vector scan,
  which is what DuckDB does without the `vss` extension.
* Embedding dimension: 768. Viewer rebuilds default to 0.5 m patches; the CLI
  retains its 1 m default unless a size is supplied.
