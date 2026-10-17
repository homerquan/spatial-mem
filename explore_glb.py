"""Explore a GLB file's scene graph, meshes, accessors, and transforms."""
import sys
import numpy as np
from pygltflib import GLTF2

GLB = sys.argv[1] if len(sys.argv) > 1 else "scenes/sample-room/sample-room.glb"

gltf = GLTF2().load(GLB)

ATTR_MAP = {
    "POSITION": "POSITION",
    "NORMAL": "NORMAL",
    "TEXCOORD_0": "TEXCOORD_0",
}

print("=== Meshes ===")
for i, m in enumerate(gltf.meshes or []):
    print(f"mesh {i}: name={m.name!r} primitives={len(m.primitives)}")
    for p in m.primitives:
        attrs = p.attributes
        def get(attr):
            return getattr(attrs, attr, None)
        print("   prim attributes POSITION=", get("POSITION"), "NORMAL=", get("NORMAL"),
              "indices=", p.indices, "material=", p.material)

print("\n=== Accessors (all) ===")
for i, a in enumerate(gltf.accessors or []):
    print(f"acc {i}: type={a.type} componentType={a.componentType} count={a.count} min={a.min} max={a.max} name={a.name!r}")

print("\n=== Materials (first 10) ===")
for i, mat in enumerate((gltf.materials or [])[:10]):
    print(f"mat {i}: name={mat.name!r}")

print("\n=== Buffers ===")
for i, b in enumerate(gltf.buffers or []):
    print(f"buff {i}: byteLength={b.byteLength} uri={b.uri!r}")

print("\n=== BufferViews ===")
for i, bv in enumerate(gltf.bufferViews or []):
    print(f"bv {i}: buffer={bv.buffer} byteOffset={bv.byteOffset} byteLength={bv.byteLength} target={bv.target}")

# Read position data of mesh 0 primitive 0
print("\n=== Read mesh 0 prim 0 positions ===")
m0 = gltf.meshes[0]
p0 = m0.primitives[0]
pos_acc = gltf.accessors[p0.attributes.POSITION]
norm_acc = gltf.accessors[p0.attributes.NORMAL] if p0.attributes.NORMAL is not None else None
print("pos accessor:", pos_acc.type, pos_acc.componentType, pos_acc.count, pos_acc.min, pos_acc.max)
print("norm accessor:", None if norm_acc is None else (norm_acc.type, norm_acc.count))

def read_accessor(acc):
    bv = gltf.bufferViews[acc.bufferView]
    buf = gltf.buffers[bv.buffer]
    # GLB: data is in the binary chunk
    blob = gltf.binary_blob()
    offset = bv.byteOffset + (acc.byteOffset or 0)
    n = acc.count
    if acc.type == "VEC3":
        comps = 3
    elif acc.type == "VEC2":
        comps = 2
    elif acc.type == "SCALAR":
        comps = 1
    else:
        comps = 1
    ct = acc.componentType  # 5126 = float
    dtype = np.float32 if ct == 5126 else np.uint16 if ct == 5123 else np.uint8
    itemsize = np.dtype(dtype).itemsize * comps
    raw = blob[offset: offset + n * itemsize]
    arr = np.frombuffer(raw, dtype=dtype).reshape(n, comps)
    return arr

pos = read_accessor(pos_acc)
print("positions shape:", pos.shape, "min:", pos.min(axis=0), "max:", pos.max(axis=0))
print("first 3 positions:\n", pos[:3])
if norm_acc is not None:
    norm = read_accessor(norm_acc)
    print("normals shape:", norm.shape, "first 3:\n", norm[:3])

# indices
if p0.indices is not None:
    idx_acc = gltf.accessors[p0.indices]
    idx = read_accessor(idx_acc).ravel()
    print("indices shape:", idx.shape, "first 10:", idx[:10])
