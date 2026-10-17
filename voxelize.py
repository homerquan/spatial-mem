"""Voxelize a parsed GLB scene into small spatial units with text descriptions.

Each occupied voxel becomes one "unit" of spatial memory:
  - 3D location (voxel center)
  - bounding box
  - dominant surface orientation (normal) -> surface type: floor/ceiling/wall/window/object/surface
  - a short text description that gets embedded
"""

from __future__ import annotations

import math
from itertools import product
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from parse_glb import Scene, Triangle, load_scene
from geometry import clip_triangle


@dataclass
class Unit:
    unit_id: str
    voxel: tuple  # (ix, iy, iz)
    center: np.ndarray  # (3,) world-space center
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    surface_type: str  # floor | ceiling | wall | window | object | surface
    facing: str  # e.g. "+x", "-y", "up", "down", "mixed"
    normal: np.ndarray  # dominant unit normal (3,)
    triangle_count: int
    area: float  # total triangle area in voxel
    description: str
    node_names: List[str] = field(default_factory=list)
    cell_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cell_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: int = 0


def _triangle_area(v0, v1, v2) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(v1 - v0, v2 - v0)))


def _surface_type_and_facing(normal: np.ndarray, z: float, zmin: float,
                             zmax: float, name_semantic: str) -> tuple:
    """Classify a surface from its dominant normal and height."""
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    az = abs(nz)
    # Name-based hints take precedence for windows
    if name_semantic == "window":
        return "window", _facing(nx, ny, nz)
    if az > 0.75:
        # Horizontal surface. Use height + normal direction.
        if nz > 0:
            # faces up
            if z < zmin + (zmax - zmin) * 0.25:
                return "floor", "up"
            return "surface", "up"
        else:
            # faces down
            if z > zmax - (zmax - zmin) * 0.25:
                return "ceiling", "down"
            return "surface", "down"
    if az < 0.30:
        return "wall", _facing(nx, ny, nz)
    return "object", _facing(nx, ny, nz)


def _facing(nx: float, ny: float, nz: float) -> str:
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if az >= ax and az >= ay:
        return "up" if nz > 0 else "down"
    if ax >= ay:
        return "+x" if nx > 0 else "-x"
    return "+y" if ny > 0 else "-y"


def _describe(unit: Unit) -> str:
    st = unit.surface_type
    if st == "wall":
        return f"wall: a vertical wall surface facing {unit.facing}"
    if st == "floor":
        return "floor: a horizontal floor surface facing up"
    if st == "ceiling":
        return "ceiling: an overhead ceiling surface facing down"
    if st == "window":
        return f"window: a window surface facing {unit.facing}"
    if st == "surface":
        return f"surface: a horizontal surface facing {unit.facing}"
    return f"object: an unlabeled object surface facing {unit.facing}"


def voxelize(scene: Scene, voxel_size: float = 1.0,
             min_area: float = 0.0, on_progress=None) -> List[Unit]:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    vs, ns, metas = scene.as_arrays()
    if len(vs) == 0:
        return []

    zmin, zmax = scene.bounds_min[2], scene.bounds_max[2]

    origin = scene.bounds_min
    triangle_min, triangle_max = vs.min(axis=1), vs.max(axis=1)
    first_cells = np.floor((triangle_min-origin)/voxel_size).astype(np.int64)
    last_cells = np.floor((triangle_max-origin)/voxel_size).astype(np.int64)
    original_areas = np.linalg.norm(np.cross(vs[:,1]-vs[:,0], vs[:,2]-vs[:,0]), axis=1)*0.5
    face_normals = np.cross(vs[:,1]-vs[:,0], vs[:,2]-vs[:,0])
    ns = face_normals / np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    centroids = vs.mean(axis=1)
    groups = {}

    def add_piece(key, triangle_index, lower, upper, area, center):
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"area": 0.0, "center_sum": np.zeros(3),
                "normal_sum": np.zeros(3), "bbox_min": lower.copy(),
                "bbox_max": upper.copy(), "semantics": {}, "nodes": [], "count": 0}
        g["area"] += area
        g["center_sum"] += center*area
        g["normal_sum"] += ns[triangle_index]*area
        np.minimum(g["bbox_min"], lower, out=g["bbox_min"])
        np.maximum(g["bbox_max"], upper, out=g["bbox_max"])
        semantic = metas[triangle_index]["semantic"]
        g["semantics"][semantic] = g["semantics"].get(semantic, 0)+area
        name = metas[triangle_index]["node_name"]
        if len(g["nodes"])<5 and name not in g["nodes"]:
            g["nodes"].append(name)
        g["count"] += 1

    for i in range(len(vs)):
        if on_progress and i % 10000 == 0:
            on_progress(i/len(vs))
        if original_areas[i] <= 1e-12:
            continue
        # Keep perpendicular surfaces separate inside a cell. A wall patch
        # should not acquire the floor's full volume or its averaged normal.
        axis = int(np.argmax(np.abs(ns[i])))
        orientation = axis*2 + int(ns[i,axis]<0)
        if np.array_equal(first_cells[i], last_cells[i]):
            key = (*map(int, first_cells[i]), orientation)
            add_piece(key, i, triangle_min[i], triangle_max[i], original_areas[i], centroids[i])
            continue
        # A triangle may cross several cells. Index every clipped piece,
        # rather than assigning the whole triangle to its centroid's cell.
        ranges = [range(int(a), int(b)+1) for a,b in zip(first_cells[i], last_cells[i])]
        for cell in product(*ranges):
            cell_min = origin + np.asarray(cell)*voxel_size
            for piece in clip_triangle(vs[i], cell_min, cell_min+voxel_size):
                area = _triangle_area(*piece)
                if area > 1e-12:
                    add_piece((*cell, orientation), i, piece.min(axis=0), piece.max(axis=0), area, piece.mean(axis=0))

    units: List[Unit] = []
    counter = 0
    for key, g in groups.items():
        if g["area"] < min_area:
            continue
        center = g["center_sum"]/g["area"]
        cell = key[:3]
        cell_min = origin + np.asarray(cell)*voxel_size
        # dominant normal weighted by area
        nsum = g["normal_sum"]
        nlen = np.linalg.norm(nsum)
        normal = nsum / nlen if nlen > 0 else np.array([0.0, 0.0, 1.0])
        # pick dominant semantic by area
        name_semantic = max(g["semantics"], key=g["semantics"].get)

        surface_type, facing = _surface_type_and_facing(
            normal, float(center[2]), zmin, zmax, name_semantic)

        unit = Unit(
            unit_id=f"u{counter:05d}",
            voxel=cell,
            center=center,
            bbox_min=g["bbox_min"],
            bbox_max=g["bbox_max"],
            surface_type=surface_type,
            facing=facing,
            normal=normal,
            triangle_count=g["count"],
            area=float(g["area"]),
            description="",  # filled below
            node_names=g["nodes"],
            cell_min=cell_min,
            cell_max=cell_min+voxel_size,
            orientation=key[3],
        )
        unit.description = _describe(unit)
        units.append(unit)
        counter += 1

    # Sort for stable ordering
    units.sort(key=lambda u: (u.voxel[2], u.voxel[1], u.voxel[0], u.facing))
    for i, u in enumerate(units):
        u.unit_id = f"u{i:05d}"
    if on_progress:
        on_progress(1.0)
    return units


def main():
    glb = "scenes/sample-room/sample-room.glb"
    scene = load_scene(glb)
    units = voxelize(scene, voxel_size=1.0)
    print(f"units: {len(units)}")
    from collections import Counter
    print("surface types:", dict(Counter(u.surface_type for u in units)))
    print("facings:", dict(Counter(u.facing for u in units)))
    for u in units[:8]:
        print(u.unit_id, u.voxel, np.round(u.center, 2), u.surface_type,
              u.facing, round(u.area, 2), "|", u.description)


if __name__ == "__main__":
    main()
