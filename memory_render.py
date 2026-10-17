"""Deterministic textured RGB/depth observations in the scan's Z-up world frame.

The rasterizer records the source triangle for every pixel. No screenshot-to-3D
registration or guessed coordinate transform is involved.
"""
import math

import numpy as np
from numba import njit


def camera_basis(direction):
    forward = np.asarray(direction, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.array([right, up, forward])


@njit(cache=True)
def _render(vertices, uvs, materials, double_sided, pixels, offsets, widths, heights,
            factors, eye, basis, size, fov, near, far):
    rgb = np.full((size, size, 3), 35, dtype=np.uint8)
    depth = np.full((size, size), np.inf, dtype=np.float32)
    owners = np.full((size, size), -1, dtype=np.int32)
    focal = size / (2 * math.tan(fov * math.pi / 360))
    for ti in range(len(vertices)):
        mat = materials[ti]
        edge1 = vertices[ti,1] - vertices[ti,0]
        edge2 = vertices[ti,2] - vertices[ti,0]
        normal = np.cross(edge1, edge2)
        if not double_sided[mat] and np.dot(normal, eye-vertices[ti,0]) <= 0:
            continue
        # Clip at the near plane, interpolating UVs with the clipped vertices.
        original = np.empty((3,5), dtype=np.float64)
        for j in range(3):
            delta = vertices[ti,j]-eye
            for axis in range(3):
                original[j,axis] = np.dot(basis[axis], delta)
            original[j,3:] = uvs[ti,j]
        clipped = np.empty((4,5), dtype=np.float64)
        count = 0
        previous = original[2]
        for j in range(3):
            current = original[j]
            if (previous[2]>=near) != (current[2]>=near):
                weight = (near-previous[2])/(current[2]-previous[2])
                clipped[count] = previous + weight*(current-previous)
                count += 1
            if current[2]>=near:
                clipped[count] = current
                count += 1
            previous = current
        for part in range(1,count-1):
            tri = np.empty((3,5), dtype=np.float64)
            tri[0] = clipped[0]; tri[1] = clipped[part]; tri[2] = clipped[part+1]
            screen = np.empty((3,2), dtype=np.float64)
            for j in range(3):
                screen[j,0] = size/2 + tri[j,0]*focal/tri[j,2]
                screen[j,1] = size/2 - tri[j,1]*focal/tri[j,2]
            x0,y0 = screen[0]; x1,y1 = screen[1]; x2,y2 = screen[2]
            determinant = (y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
            if abs(determinant) < 1e-8:
                continue
            lo_x = max(0,int(math.floor(min(x0,x1,x2))))
            hi_x = min(size-1,int(math.ceil(max(x0,x1,x2))))
            lo_y = max(0,int(math.floor(min(y0,y1,y2))))
            hi_y = min(size-1,int(math.ceil(max(y0,y1,y2))))
            for y in range(lo_y,hi_y+1):
                for x in range(lo_x,hi_x+1):
                    a = ((y1-y2)*(x+.5-x2)+(x2-x1)*(y+.5-y2))/determinant
                    b = ((y2-y0)*(x+.5-x2)+(x0-x2)*(y+.5-y2))/determinant
                    c = 1-a-b
                    if min(a,b,c) < -1e-8:
                        continue
                    inv_z = a/tri[0,2]+b/tri[1,2]+c/tri[2,2]
                    z = 1/inv_z
                    if z >= depth[y,x] or z>far:
                        continue
                    depth[y,x] = z
                    owners[y,x] = ti
                    if widths[mat]:
                        u = (a*tri[0,3]/tri[0,2]+b*tri[1,3]/tri[1,2]+c*tri[2,3]/tri[2,2])*z
                        v = (a*tri[0,4]/tri[0,2]+b*tri[1,4]/tri[1,2]+c*tri[2,4]/tri[2,2])*z
                        # glTF UV origin and embedded images are both top-left.
                        tx = int((u%1)*widths[mat])%widths[mat]
                        ty = int((v%1)*heights[mat])%heights[mat]
                        offset = offsets[mat]+(ty*widths[mat]+tx)*3
                        for channel in range(3):
                            rgb[y,x,channel] = min(255,pixels[offset+channel]*factors[mat,channel])
                    else:
                        for channel in range(3):
                            rgb[y,x,channel] = min(255,200*factors[mat,channel])
    return rgb, depth, owners


class ScanRenderer:
    def __init__(self, scene):
        self.scene = scene
        self.vertices = np.asarray([[t.v0,t.v1,t.v2] for t in scene.triangles], dtype=np.float64)
        self.uvs = np.asarray([t.uv if t.uv is not None else np.zeros((3,2)) for t in scene.triangles])
        self.normals = np.cross(self.vertices[:,1]-self.vertices[:,0], self.vertices[:,2]-self.vertices[:,0])
        self.normals /= np.maximum(np.linalg.norm(self.normals,axis=1,keepdims=True),1e-12)
        self.orientations = np.argmax(np.abs(self.normals),axis=1)*2
        self.orientations += (self.normals[np.arange(len(self.normals)),self.orientations//2]<0)
        fallback = {'image':None,'double_sided':False,'factor':[1,1,1]}
        materials = scene.materials+[fallback]
        self.materials = np.asarray([t.material_index if t.material_index>=0 else len(materials)-1 for t in scene.triangles])
        self.double_sided = np.asarray([m['double_sided'] for m in materials])
        self.factors = np.asarray([m['factor'] for m in materials],dtype=np.float64)
        self.widths = np.asarray([m['image'].shape[1] if m['image'] is not None else 0 for m in materials])
        self.heights = np.asarray([m['image'].shape[0] if m['image'] is not None else 0 for m in materials])
        lengths = self.widths*self.heights*3
        self.offsets = np.r_[0,np.cumsum(lengths)[:-1]]
        self.pixels = np.concatenate([m['image'].reshape(-1) for m in materials if m['image'] is not None] + [np.zeros(1,dtype=np.uint8)])

    def render(self, position, direction, size=256, fov=70):
        basis = camera_basis(direction)
        rgb,depth,owners = _render(self.vertices,self.uvs,self.materials,self.double_sided,
            self.pixels,self.offsets,self.widths,self.heights,self.factors,
            np.asarray(position,dtype=np.float64),basis,size,fov,.08,15.)
        return rgb,depth,owners,basis

    def walkthrough(self, spacing=2.0, eye_height=1.6):
        """Sample supported floor locations, with headroom and wall clearance."""
        lower,upper = self.scene.bounds_min,self.scene.bounds_max
        v = self.vertices
        floor_indices = np.flatnonzero(self.normals[:,2]>.8)
        floors = v[floor_indices]
        if not len(floors):
            raise ValueError('No upward-facing floor was found for an eye-level walkthrough.')
        # Area-weighted levels suppress small horizontal furniture tops.
        areas = np.linalg.norm(np.cross(floors[:,1]-floors[:,0],floors[:,2]-floors[:,0]),axis=1)/2
        levels = np.floor(floors.mean(axis=1)[:,2]/.2).astype(int)
        level_area = {int(level):float(areas[levels==level].sum()) for level in np.unique(levels)}
        threshold = max(level_area.values())*.08
        allowed = {level for level,area in level_area.items() if area>=threshold}
        poses = []
        spacing = max(spacing, math.sqrt(max(1,(upper[0]-lower[0])*(upper[1]-lower[1]))/60))
        for x in np.arange(lower[0]+spacing/2,upper[0],spacing):
            for y in np.arange(lower[1]+spacing/2,upper[1],spacing):
                hits = self._vertical_hits(x,y)
                candidates = [z for z,ti in hits if self.normals[ti,2]>.8 and int(math.floor(z/.2)) in allowed]
                accepted = []
                for z in candidates:
                    if any(abs(z-old)<.3 for old in accepted): continue
                    if any(z+.12<h<z+eye_height+.2 for h,_ in hits): continue
                    eye = np.array([x,y,z+eye_height])
                    if not self._clear(eye,.3): continue
                    accepted.append(z)
                    for heading in range(8):
                        yaw = heading*math.pi/4
                        direction = np.array([math.cos(yaw),math.sin(yaw),-.06])
                        direction /= np.linalg.norm(direction)
                        poses.append((eye.copy(),direction))
        if not poses:
            raise ValueError('No camera location has sufficient floor support and headroom.')
        return poses

    def _vertical_hits(self,x,y):
        v = self.vertices
        a,b,c = v[:,0],v[:,1],v[:,2]
        determinant = (b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
        safe = np.where(np.abs(determinant)>1e-12,determinant,np.inf)
        wa = ((b[:,1]-c[:,1])*(x-c[:,0])+(c[:,0]-b[:,0])*(y-c[:,1]))/safe
        wb = ((c[:,1]-a[:,1])*(x-c[:,0])+(a[:,0]-c[:,0])*(y-c[:,1]))/safe
        wc = 1-wa-wb
        ids = np.flatnonzero((np.abs(determinant)>1e-12)&(wa>=0)&(wb>=0)&(wc>=0))
        z = wa*a[:,2]+wb*b[:,2]+wc*c[:,2]
        return sorted((float(z[i]),int(i)) for i in ids)

    def _clear(self,eye,distance):
        # Point-to-triangle distance, evaluated only in the local AABB.
        v = self.vertices
        local = v[(v.min(axis=1)<=eye+distance).all(axis=1)&(v.max(axis=1)>=eye-distance).all(axis=1)]
        if not len(local): return True
        for tri in local:
            a,b,c = tri
            normal = np.cross(b-a,c-a)
            length = np.linalg.norm(normal)
            if length<1e-12: continue
            normal /= length
            plane_distance = np.dot(eye-a,normal)
            projected = eye-plane_distance*normal
            signs = [np.dot(np.cross(tri[(i+1)%3]-tri[i],projected-tri[i]),normal) for i in range(3)]
            if min(signs)>=-1e-9 and abs(plane_distance)<distance: return False
            for i in range(3):
                start,end = tri[i],tri[(i+1)%3]
                edge = end-start
                t = np.clip(np.dot(eye-start,edge)/max(np.dot(edge,edge),1e-12),0,1)
                if np.linalg.norm(eye-(start+t*edge))<distance: return False
        return True


def unproject(depth, position, basis, fov):
    size = depth.shape[0]
    yy,xx = np.mgrid[:size,:size]
    scale = 2*math.tan(math.radians(fov)/2)/size
    directions = basis[2]+((xx+.5-size/2)*scale)[...,None]*basis[0]-((yy+.5-size/2)*scale)[...,None]*basis[1]
    return np.asarray(position)+directions*depth[...,None]
