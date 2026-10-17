"""Parse a GLB file into world-space triangles with normals and semantic labels.

The HM3D-style GLBs used here store each surface as a separate mesh node whose
name encodes semantics, e.g.:

    chunk000_group000_sub007_objectceiling010_n+0+0-100_type002

Nodes have identity transforms in these files, so vertex positions are already
in world space, but we still apply node transforms generically.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
from pygltflib import GLTF2

COMPONENT_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                    5123: np.uint16, 5125: np.uint32, 5126: np.float32}
VEC_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
                  "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass
class Triangle:
    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    normal: np.ndarray
    mesh_name: str
    node_name: str
    semantic: str
    material_index: int
    uv: Optional[np.ndarray] = None


@dataclass
class Scene:
    triangles: List[Triangle] = field(default_factory=list)
    bounds_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bounds_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gltf_path: str = ""
    surface_area: float = 0.0
    materials: list = field(default_factory=list)

    def as_arrays(self):
        if not self.triangles:
            empty = np.zeros((0, 3, 3))
            return empty, np.zeros((0, 3)), []
        vs = np.array([[t.v0, t.v1, t.v2] for t in self.triangles])
        ns = np.array([t.normal for t in self.triangles])
        metas = [{"mesh_name": t.mesh_name, "node_name": t.node_name,
                  "semantic": t.semantic, "material_index": t.material_index}
                 for t in self.triangles]
        return vs, ns, metas


def _read_accessor(gltf: GLTF2, acc) -> np.ndarray:
    if acc.bufferView is None or acc.sparse is not None:
        raise ValueError("Sparse or missing-buffer accessors are not supported by this indexer")
    bv = gltf.bufferViews[acc.bufferView]
    blob = gltf.binary_blob()
    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    n = acc.count
    comps = VEC_COMPONENTS.get(acc.type, 1)
    dtype = COMPONENT_DTYPES.get(acc.componentType, np.float32)
    itemsize = np.dtype(dtype).itemsize * comps
    arr = np.ndarray((n, comps), dtype=dtype, buffer=blob, offset=offset,
                     strides=(bv.byteStride or itemsize, np.dtype(dtype).itemsize))
    if acc.normalized and np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = arr.astype(np.float64)/info.max
        if info.min < 0:
            arr = np.maximum(arr, -1)
    return arr


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _node_transform(node) -> np.ndarray:
    """Return a glTF local transform; matrices are stored column-major."""
    if node.matrix is not None:
        return np.asarray(node.matrix, dtype=np.float64).reshape(4, 4, order="F")
    T = np.eye(4)
    if node.translation:
        T[:3, 3] = np.array(node.translation, dtype=np.float64)
    if node.rotation is not None:
        q = np.array(node.rotation, dtype=np.float64)  # (x, y, z, w)
        R = _quat_matrix(q)
        T[:3, :3] = R
    if node.scale:
        T[:3, :3] *= np.array(node.scale, dtype=np.float64)[None, :]
    return T


NORMAL_NAME_RE = re.compile(r"n([+-]\d+)([+-]\d+)([+-]\d+)")


def _normal_from_name(name: str) -> Optional[np.ndarray]:
    m = NORMAL_NAME_RE.search(name)
    if not m:
        return None
    vals = [int(g) / 100.0 for g in m.groups()]
    n = np.array(vals, dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm == 0:
        return None
    return n / norm


def _semantic_from_name(name: str) -> str:
    low = name.lower()
    if "objectfloor" in low or "_floor" in low:
        return "floor"
    if "objectceiling" in low or "_ceiling" in low:
        return "ceiling"
    if "objectwall" in low or "_wall" in low:
        return "wall"
    if "window" in low:
        return "window"
    if "door" in low:
        return "door"
    if "object" in low:
        return "object"
    return "unknown"


def load_scene(glb_path: str) -> Scene:
    gltf = GLTF2().load(glb_path)
    scene = Scene(gltf_path=glb_path)
    # Keep UV coordinates and base-color textures for eye-level observations.
    # Geometry and visual memories share these exact world-space triangles.
    from PIL import Image
    images = {}
    for index, image in enumerate(gltf.images or []):
        if image.bufferView is not None:
            view = gltf.bufferViews[image.bufferView]
            offset = view.byteOffset or 0
            data = gltf.binary_blob()[offset:offset+view.byteLength]
        elif image.uri and not image.uri.startswith('data:'):
            root = Path(glb_path).resolve().parent
            image_path = (root / image.uri).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError('Texture path must stay within the scene directory')
            data = image_path.read_bytes()
        elif image.uri and image.uri.startswith('data:'):
            import base64
            data = base64.b64decode(image.uri.split(',', 1)[1])
        else:
            continue
        with Image.open(BytesIO(data)) as texture:
            images[index] = np.asarray(texture.convert('RGB'))
    for material in gltf.materials or []:
        pbr = material.pbrMetallicRoughness
        texture_info = pbr.baseColorTexture if pbr else None
        texture = gltf.textures[texture_info.index] if texture_info else None
        transform = (texture_info.extensions or {}).get('KHR_texture_transform', {}) if texture_info else {}
        scene.materials.append({'image': images.get(texture.source) if texture else None,
            'factor': (pbr.baseColorFactor or [1,1,1,1])[:3] if pbr else [1,1,1],
            'double_sided': bool(material.doubleSided), 'uv_transform': transform,
            'texcoord': transform.get('texCoord', texture_info.texCoord or 0) if texture_info else 0})

    if gltf.scenes:
        scene_nodes = gltf.scenes[gltf.scene or 0].nodes or []
    else:
        children = {child for node in gltf.nodes or [] for child in node.children or []}
        scene_nodes = [i for i in range(len(gltf.nodes or [])) if i not in children]

    def visit(node_idx, parent_transform, ancestors):
        if node_idx in ancestors:
            raise ValueError("Cycle in GLB node hierarchy")
        node = gltf.nodes[node_idx]
        transform = parent_transform @ _node_transform(node)
        if node.mesh is not None:
            read_mesh(node_idx, node, transform)
        for child in node.children or []:
            visit(child, transform, ancestors | {node_idx})

    def read_mesh(node_idx, node, transform):
        R = transform[:3, :3]
        t = transform[:3, 3]
        mesh = gltf.meshes[node.mesh]
        node_name = node.name or f"node{node_idx}"
        semantic = _semantic_from_name(node_name)

        for prim in mesh.primitives:
            if prim.mode not in (None, 4):
                raise ValueError(f"Non-triangle primitive in {node_name}; indexing would omit geometry")
            if prim.attributes is None or prim.attributes.POSITION is None:
                continue
            pos_acc = gltf.accessors[prim.attributes.POSITION]
            pos = _read_accessor(gltf, pos_acc).astype(np.float64)  # (V,3)
            # Apply world transform
            pos_w = pos @ R.T + t
            material_index = int(prim.material) if prim.material is not None else -1
            material = scene.materials[material_index] if material_index >= 0 else None
            uv_accessor = getattr(prim.attributes, f'TEXCOORD_{material["texcoord"] if material else 0}', None)
            uv = _read_accessor(gltf, gltf.accessors[uv_accessor]).astype(np.float64) if uv_accessor is not None else None
            if uv is not None and material:
                uv_transform = material['uv_transform']
                uv = uv * np.asarray(uv_transform.get('scale', [1,1]))
                angle = uv_transform.get('rotation', 0)
                uv = uv @ np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
                uv += np.asarray(uv_transform.get('offset', [0,0]))

            normals: Optional[np.ndarray] = None
            if prim.attributes.NORMAL is not None:
                norm_acc = gltf.accessors[prim.attributes.NORMAL]
                normals = _read_accessor(gltf, norm_acc).astype(np.float64)  # (V,3)
                normals = normals @ np.linalg.pinv(R)
                nlen = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = np.where(nlen > 0, normals / np.maximum(nlen, 1e-12), normals)

            if prim.indices is not None:
                idx = _read_accessor(gltf, gltf.accessors[prim.indices]).ravel().astype(np.int64)
                verts = pos_w[idx]  # (M,3)
            else:
                verts = pos_w
            triangle_uv = (uv[idx] if prim.indices is not None else uv) if uv is not None else None

            if normals is not None:
                if prim.indices is not None:
                    vnorm = normals[idx]
                else:
                    vnorm = normals
            else:
                vnorm = None

            # Iterate triangles
            n_tri = len(verts) // 3
            for tri_i in range(n_tri):
                v0, v1, v2 = verts[tri_i * 3], verts[tri_i * 3 + 1], verts[tri_i * 3 + 2]
                if vnorm is not None:
                    n = vnorm[tri_i * 3]
                else:
                    e1 = v1 - v0
                    e2 = v2 - v0
                    n = np.cross(e1, e2)
                    ln = np.linalg.norm(n)
                    n = n / ln if ln > 0 else np.array([0.0, 0.0, 1.0])
                scene.triangles.append(Triangle(
                    v0=v0, v1=v1, v2=v2, normal=n,
                    mesh_name=mesh.name or f"mesh{node.mesh}",
                    node_name=node_name,
                    semantic=semantic,
                    material_index=material_index,
                    uv=triangle_uv[tri_i*3:tri_i*3+3] if triangle_uv is not None else None,
                ))

    for node_idx in scene_nodes:
        visit(node_idx, np.eye(4), set())

    if scene.triangles:
        vs, _, _ = scene.as_arrays()
        scene.bounds_min = vs.reshape(-1, 3).min(axis=0)
        scene.bounds_max = vs.reshape(-1, 3).max(axis=0)
        scene.surface_area = float(np.linalg.norm(np.cross(vs[:,1]-vs[:,0], vs[:,2]-vs[:,0]), axis=1).sum()*0.5)
    return scene


def main():
    glb_path = sys.argv[1] if len(sys.argv) > 1 else "scenes/sample-room/sample-room.glb"
    scene = load_scene(glb_path)
    vs, ns, metas = scene.as_arrays()
    print(f"GLB: {glb_path}")
    print(f"triangles: {len(scene.triangles)}")
    print(f"bounds min: {scene.bounds_min} max: {scene.bounds_max} size: {scene.bounds_max - scene.bounds_min}")
    from collections import Counter
    sem = Counter(m["semantic"] for m in metas)
    print("semantics:", dict(sem))
    # normal stats
    nn = ns / (np.linalg.norm(ns, axis=1, keepdims=True) + 1e-12)
    print("normal z mean:", float(nn[:, 2].mean()), "abs z mean:", float(np.abs(nn[:, 2]).mean()))


if __name__ == "__main__":
    main()
