"""Local image-text retrieval over registered first-person observations."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading

import numpy as np
from PIL import Image

MODEL_NAME = 'google/siglip2-base-patch16-224'
MODEL_REVISION = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'


class VisualEncoder:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.model = self.processor = None

    def _load(self):
        if self.model is not None: return
        import torch
        from transformers import AutoModel, AutoProcessor
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        kwargs = dict(cache_dir=str(self.root/'.models'), revision=MODEL_REVISION)
        processor = AutoProcessor.from_pretrained(MODEL_NAME, **kwargs)
        model = AutoModel.from_pretrained(MODEL_NAME, **kwargs).eval().to(self.device)
        self.processor, self.model = processor, model

    def encode(self, images=None, text=None):
        import torch
        with self.lock, torch.inference_mode():
            self._load()
            if images is not None:
                inputs = self.processor(images=images, return_tensors='pt').to(self.device)
                features = self.model.get_image_features(**inputs)
            else:
                inputs = self.processor(text=[f'This is a photo of {text}.'],
                    padding='max_length', truncation=True, max_length=64,
                    return_tensors='pt').to(self.device)
                features = self.model.get_text_features(**inputs)
            # Transformers 5 returns a pooling output; earlier versions return a tensor.
            if not isinstance(features, torch.Tensor): features = features.pooler_output
            features = features / features.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            return features.cpu().numpy().astype(np.float32)


def pose(position, direction, target=None, fov=70):
    direction = np.asarray(direction,dtype=float)
    direction /= np.linalg.norm(direction)
    return {'position':np.asarray(position).tolist(), 'direction':direction.tolist(),
        'target':np.asarray(target if target is not None else np.asarray(position)+direction*3).tolist(),
        'up':[0,0,1], 'fov':float(fov),
        'yaw':math.degrees(math.atan2(direction[1],direction[0])),
        'pitch':math.degrees(math.asin(np.clip(direction[2],-1,1)))}


@dataclass
class VisualMemory:
    records: list
    embeddings: np.ndarray
    directory: Path

    @classmethod
    def read(cls, directory):
        directory = Path(directory)
        data = json.loads((directory/'observations.json').read_text())
        return cls(data['records'],np.load(directory/'visual.npy',allow_pickle=False),directory)

    def search(self, query, encoder, k):
        scores = self.embeddings @ encoder.encode(text=query)[0]
        order = np.argsort(-scores,kind='stable')
        selected = []
        # Suppress overlapping crops of the same view and duplicate nearby views
        # of the same location, while preserving distinct objects in a room.
        for index in order:
            record = self.records[int(index)]
            center = np.asarray(record['center'])
            if any((record['view_id']==old['view_id'] and self._overlap(record['crop'],old['crop'])>.35)
                   or (np.linalg.norm(center-np.asarray(old['center']))<.7 and
                       np.dot(record['camera']['direction'],old['camera']['direction'])>.5)
                   for old in selected): continue
            selected.append({**record,'similarity':float(scores[index]),'match_kind':'visual'})
            if len(selected)>=k: break
        return selected

    @staticmethod
    def _overlap(a,b):
        intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
        smaller = min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
        return intersection/max(smaller,1e-9)


def build_visual_memory(scene, units, directory, encoder, on_progress=None):
    from memory_render import ScanRenderer, unproject
    directory = Path(directory)
    images_dir = directory/'observations'
    images_dir.mkdir(parents=True,exist_ok=True)
    frames_dir = directory/'frames'
    frames_dir.mkdir(exist_ok=True)
    renderer = ScanRenderer(scene)
    path = renderer.walkthrough()
    patch_size = float(units[0].cell_max[0]-units[0].cell_min[0])
    lookup = {(*unit.voxel,unit.orientation):i for i,unit in enumerate(units)}
    records, embeddings, batch, pending = [], [], [], []
    best_views = {}
    coverage = set()
    frame_poses = []
    size,fov = 256,70
    crops = [(0,0,1,1)]+[(x,y,x+.5,y+.5) for y in (0,.25,.5) for x in (0,.25,.5)]

    def flush():
        if batch:
            embeddings.append(encoder.encode(images=batch))
            records.extend(pending)
            batch.clear(); pending.clear()

    for view_index,(eye,direction) in enumerate(path):
        if on_progress: on_progress(view_index/len(path),f'Walking view {view_index+1}/{len(path)} · {len(records)+len(pending):,} observations')
        rgb,depth,owners,basis = renderer.render(eye,direction,size,fov)
        valid = owners>=0
        if valid.mean()<.12: continue
        # Only unproject valid pixels. Empty/background pixels never acquire coordinates.
        points = unproject(np.where(valid,depth,0),eye,basis,fov)
        cells = np.floor((points-scene.bounds_min)/patch_size).astype(np.int64)
        pixel_patches = np.full((size,size),-1,dtype=np.int32)
        yy,xx = np.nonzero(valid)
        keys = np.column_stack((cells[yy,xx],renderer.orientations[owners[yy,xx]]))
        unique,inverse = np.unique(keys,axis=0,return_inverse=True)
        ids = np.asarray([lookup.get(tuple(key),-1) for key in unique])
        pixel_patches[yy,xx] = ids[inverse]
        image = Image.fromarray(rgb)
        base_pose = pose(eye,direction,fov=fov)
        image.save(frames_dir/f'{view_index:04d}.jpg',quality=90)
        np.savez_compressed(frames_dir/f'{view_index:04d}.npz',depth=depth,
            triangle_ids=owners,patch_indices=pixel_patches)
        frame_poses.append({'view_id':view_index,'camera':base_pose})
        flat_patches = pixel_patches[valid]
        flat_points = points[valid]
        order = np.argsort(flat_patches)
        sorted_patches = flat_patches[order]
        patch_ids,starts,counts = np.unique(sorted_patches,return_index=True,return_counts=True)
        for unit_index,start,count in zip(patch_ids,starts,counts):
            if unit_index<0: continue
            coverage.add(int(unit_index))
            if count > best_views.get(int(unit_index),(0,None))[0]:
                visible = flat_points[order[start:start+count]]
                target = visible[np.argmin(np.linalg.norm(visible-units[int(unit_index)].center,axis=1))]
                best_views[int(unit_index)] = (int(count),pose(eye,target-eye,target,fov=50))
        for crop_index,crop in enumerate(crops):
            x0,y0,x1,y1 = [round(value*size) for value in crop]
            mask = valid[y0:y1,x0:x1]
            if mask.mean()<.5: continue
            region_points = points[y0:y1,x0:x1][mask]
            region_depth = depth[y0:y1,x0:x1][mask]
            # Trim isolated scan fragments; keep the nearest depth layer that
            # occupies a substantial part of the observation.
            target = np.median(region_points,axis=0)
            target_pixel = np.argmin(np.linalg.norm(region_points-target,axis=1))
            target = region_points[target_pixel]
            ids,counts = np.unique(pixel_patches[y0:y1,x0:x1][mask],return_counts=True)
            keep = [(int(i),int(count)) for i,count in zip(ids,counts) if i>=0 and count>=max(3,mask.sum()*.008)]
            if not keep: continue
            keep.sort(key=lambda pair:-pair[1])
            visible_ids = [units[i].unit_id for i,_ in keep]
            to_target = target-eye
            crop_fov = math.degrees(2*math.atan(math.tan(math.radians(fov)/2)*(crop[3]-crop[1])))
            record_pose = pose(eye,to_target,target,fov=crop_fov)
            # Original capture pose plus crop are retained separately, so the
            # observation can be reproduced exactly as well as aimed at its region.
            record_id = f'v{view_index:04d}c{crop_index:02d}'
            filename = record_id+'.jpg'
            cropped = image.crop((x0,y0,x1,y1))
            cropped.save(images_dir/filename,quality=88)
            pending.append({'unit_id':record_id,'view_id':view_index,'crop':list(crop),
                'center':target.tolist(),'bbox_min':region_points.min(axis=0).tolist(),
                'bbox_max':region_points.max(axis=0).tolist(),
                'camera':record_pose,'capture_camera':base_pose,'image':filename,
                'surface_type':'observation','facing':'', 'description':'Eye-level visual memory',
                'patch_ids':visible_ids, 'visible_pixel_count':int(mask.sum()),
                'depth_range':[float(region_depth.min()),float(region_depth.max())]})
            batch.append(cropped)
            if len(batch)>=24: flush()
    flush()
    if not records: raise ValueError('The walkthrough did not produce any registered visual observations.')
    np.save(directory/'visual.npy',np.concatenate(embeddings))
    (frames_dir/'poses.json').write_text(json.dumps(frame_poses))
    (directory/'observations.json').write_text(json.dumps({'model':MODEL_NAME,'revision':MODEL_REVISION,
        'records':records,'surface_views':{units[i].unit_id:view for i,(_,view) in best_views.items()}}))
    return {'visual_model':MODEL_NAME,'visual_revision':MODEL_REVISION,'memory_version':1,
        'observation_count':len(records),'walkthrough_view_count':len(frame_poses),
        'observed_patch_count':len(coverage),'visual_semantics':True,'retrieval_mode':'visual_walkthrough'}
