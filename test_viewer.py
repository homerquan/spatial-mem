"""Geometry, full-file rebuild, snapshot, and API regressions without model downloads."""
import importlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from geometry import clip_triangle
from index_service import IndexManager
from parse_glb import load_scene
from scene_metadata import scene_fingerprint
from voxelize import voxelize
from visual_memory import pose, VisualMemory

ROOT = Path(__file__).resolve().parent


def write_glb(path, transformed=False):
    if transformed:
        points = [[0,0,0],[1,0,0],[0,1,0]]
    else:
        points = [[0,0,0],[2,0,0],[0,2,0], [0,0,0],[0,2,0],[0,0,2]]
    # Include an interleaved padding float to catch stride/offset bugs.
    values = np.asarray([point+[99] for point in points], dtype='<f4')
    blob = values.tobytes()
    nodes = [{"mesh":0}]
    if transformed:
        matrix = np.array([[0,-1,0,10],[1,0,0,20],[0,0,1,30],[0,0,0,1]])
        nodes = [{"matrix":matrix.flatten(order='F').tolist(),"children":[1]},
                 {"mesh":0,"scale":[2,3,4]}]
    document = {"asset":{"version":"2.0"}, "scene":0,"scenes":[{"nodes":[0]}],
        "nodes":nodes,"meshes":[{"primitives":[{"attributes":{"POSITION":0},"mode":4}]}],
        "buffers":[{"byteLength":len(blob)}],
        "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":len(blob),"byteStride":16}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":len(points),"type":"VEC3"}]}
    encoded = json.dumps(document).encode()
    encoded += b' '*((-len(encoded)) % 4)
    data = struct.pack('<III',0x46546C67,2,12+8+len(encoded)+8+len(blob))
    data += struct.pack('<II',len(encoded),0x4E4F534A)+encoded
    data += struct.pack('<II',len(blob),0x004E4942)+blob
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)


class FakeModel:
    def __init__(self, *_): self.calls = 0
    def encode(self, texts, **_):
        self.calls += 1
        kinds = ['wall','floor','ceiling','window','object','surface']
        return np.asarray([np.eye(6)[next((i for i,kind in enumerate(kinds) if kind in text.lower()),4)] for text in texts])


class FakeVisualEncoder:
    def encode(self, text=None, images=None):
        return np.asarray([[0.,1.] if text=='bed' else [1.,0.]])


def fake_visual_memory(scene, units, directory, encoder, on_progress=None):
    directory=Path(directory)
    (directory/'observations').mkdir()
    records=[]
    for i,unit in enumerate(units[:2]):
        camera=pose(np.asarray(unit.center)+[0,-2,1],np.array([0,2,-1]),unit.center)
        records.append({'unit_id':f'view{i}','view_id':i,'crop':[0,0,1,1],
            'center':unit.center.tolist(),'bbox_min':unit.bbox_min.tolist(),
            'bbox_max':unit.bbox_max.tolist(),'camera':camera,'image':f'{i}.jpg',
            'surface_type':'observation','facing':'','description':'Eye-level visual memory',
            'patch_ids':[unit.unit_id]})
        Image.new('RGB',(16,16),(i*255,0,0)).save(directory/'observations'/f'{i}.jpg')
    (directory/'observations.json').write_text(json.dumps({'records':records,
        'surface_views':{unit.unit_id:records[0]['camera'] for unit in units}}))
    np.save(directory/'visual.npy',np.eye(2,dtype=np.float32))
    return {'memory_version':1,'retrieval_mode':'visual_walkthrough','visual_semantics':True,
        'observation_count':2,'walkthrough_view_count':1}


class GeometryTests(unittest.TestCase):
    def test_world_matrices_hierarchy_scale_and_interleaved_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'transformed.glb'
            write_glb(path,transformed=True)
            triangle, = load_scene(str(path)).triangles
            np.testing.assert_allclose([triangle.v0,triangle.v1,triangle.v2],
                                       [[10,20,30],[10,22,30],[7,20,30]])

    def test_triangle_crossing_cells_is_fully_indexed_and_bounds_are_fitted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.glb'; write_glb(path)
            scene = load_scene(str(path))
            units = voxelize(scene,voxel_size=0.5,min_area=0)
            self.assertGreater(len(units),2)
            self.assertAlmostEqual(sum(unit.area for unit in units),4.0)
            for unit in units:
                self.assertTrue(np.all(unit.bbox_min>=unit.cell_min-1e-9))
                self.assertTrue(np.all(unit.bbox_max<=unit.cell_max+1e-9))
                self.assertTrue(np.all(unit.center>=unit.bbox_min-1e-9))
                self.assertTrue(np.all(unit.center<=unit.bbox_max+1e-9))
                self.assertLessEqual(np.min(unit.bbox_max-unit.bbox_min),1e-9)
                self.assertNotIn('located at',unit.description)
            for cell in [(0,0,0),(2,0,0),(0,2,0)]:
                self.assertTrue(any(unit.voxel==cell for unit in units))

    def test_triangle_clipping_never_escapes_requested_bounds(self):
        triangle=np.array([[-2,0,0],[2,0,0],[0,2,0]],dtype=float)
        pieces=clip_triangle(triangle,np.array([0,0,0]),np.array([1,1,1]))
        self.assertTrue(pieces)
        for piece in pieces:
            self.assertTrue(np.all(piece>=0) and np.all(piece<=1))


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake=types.ModuleType('sentence_transformers');fake.SentenceTransformer=FakeModel
        with patch.dict(sys.modules,sentence_transformers=fake):
            cls.server=importlib.import_module('app')

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.model_path=self.root/'scenes'/'sample-room'/'different filename.glb'
        write_glb(self.model_path)
        self.manager=IndexManager(self.root,FakeModel(),visual_encoder=FakeVisualEncoder())
        self.patches=[patch.object(self.server,'SCENES_DIR',str(self.root/'scenes')),
                      patch.object(self.server,'manager',self.manager),
                      patch.object(self.server,'model',self.manager.model),
                      patch('index_service.build_visual_memory',side_effect=fake_visual_memory)]
        for item in self.patches:item.start()
        self.client=self.server.app.test_client()

    def tearDown(self):
        if self.manager.status() and self.manager.status()['status']=='running':self.wait_for_job()
        for item in reversed(self.patches):item.stop()
        self.temporary.cleanup()

    def wait_for_job(self):
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            job=self.client.get('/index/status?scene=sample-room').get_json()['job']
            if job and job['status']!='running':return job
            time.sleep(0.01)
        self.fail('Rebuild did not finish')

    def build(self):
        response=self.client.post('/reindex',json={'scene':'sample-room','voxel_size':0.5})
        self.assertEqual(response.status_code,202)
        job=self.wait_for_job();self.assertEqual(job['status'],'complete',job)
        return job

    def test_catalog_uses_actual_filename_and_ignores_empty_directories(self):
        (self.root/'scenes'/'empty').mkdir()
        entry,=self.client.get('/scenes').get_json()
        self.assertFalse(entry['indexed'])
        response=self.client.get(entry['glb_url'],headers={'Range':'bytes=0-11'})
        self.assertEqual(response.status_code,206)
        self.assertEqual(response.data[:4],b'glTF');response.close()
        self.assertIn('different%20filename.glb',entry['glb_url'])

    def test_full_file_rebuild_is_searchable_and_survives_restart(self):
        job=self.build()
        self.assertEqual(job['source_triangle_count'],2)
        entry,=self.client.get('/scenes').get_json();self.assertTrue(entry['indexed'])
        data=self.client.get('/search?q=wall&scene=sample-room&surface=wall').get_json()
        self.assertTrue(data['supported']);self.assertTrue(data['results'])
        self.assertEqual(data['generation'],job['id'])
        areas=[unit['area'] for unit in data['results']]
        self.assertEqual(areas,sorted(areas,reverse=True))
        for unit in data['results']:
            self.assertEqual(unit['surface_type'],'wall')
            np.testing.assert_allclose(unit['bbox_min'][0],0)
            np.testing.assert_allclose(unit['bbox_max'][0],0)
            self.assertGreaterEqual(unit['orientation'],0)
        restarted=IndexManager(self.root,FakeModel())
        snapshot=restarted.get(scene_fingerprint(self.model_path))
        self.assertEqual(snapshot.metadata['generation'],job['id'])
        self.assertEqual(snapshot.metadata['source_triangle_count'],2)

    def test_objects_and_wall_art_use_registered_visual_memories(self):
        self.build();calls=self.manager.model.calls
        for query,expected in [('painting on the wall','view0'),('bed','view1')]:
            data=self.client.get('/search',query_string={'q':query,'scene':'sample-room'}).get_json()
            self.assertTrue(data['supported']);self.assertEqual(data['results'][0]['unit_id'],expected)
            self.assertEqual(data['retrieval_mode'],'visual_walkthrough')
            self.assertNotIn('suggested_query',data)
            result=data['results'][0]
            self.assertEqual(result['position'],result['camera']['position'])
            self.assertEqual(set(result['view_angle']),{'yaw','pitch'})
            self.assertTrue(result['patches'])
            response=self.client.get(result['image_url'])
            self.assertEqual(response.status_code,200);response.close()
        self.assertEqual(self.manager.model.calls,calls)

    def test_visual_memory_and_pose_survive_restart(self):
        self.build()
        restarted=IndexManager(self.root,FakeModel(),visual_encoder=FakeVisualEncoder())
        snapshot=restarted.get(scene_fingerprint(self.model_path))
        self.assertTrue(snapshot.info()['visual_semantics'])
        result=snapshot.visual.search('bed',restarted.visual_encoder,1)[0]
        self.assertEqual(result['unit_id'],'view1')
        self.assertEqual(result['camera']['up'],[0,0,1])
        self.assertAlmostEqual(np.linalg.norm(result['camera']['direction']),1)

    def test_failed_visual_rebuild_preserves_memory_and_geometry(self):
        first=self.build()
        with patch('index_service.build_visual_memory',side_effect=ValueError('capture failed')):
            self.assertEqual(self.client.post('/reindex',json={'scene':'sample-room'}).status_code,202)
            self.assertEqual(self.wait_for_job()['status'],'failed')
        snapshot=self.manager.get(scene_fingerprint(self.model_path))
        self.assertEqual(snapshot.metadata['generation'],first['id'])
        self.assertTrue(snapshot.visual)

    def test_failed_rebuild_keeps_previous_complete_snapshot(self):
        first=self.build()
        with patch('index_service.load_scene',side_effect=ValueError('broken GLB')):
            response=self.client.post('/reindex',json={'scene':'sample-room'})
            self.assertEqual(response.status_code,202)
            self.assertEqual(self.wait_for_job()['status'],'failed')
        self.assertEqual(self.manager.get(scene_fingerprint(self.model_path)).metadata['generation'],first['id'])
        self.assertEqual(self.client.get('/search?q=wall&scene=sample-room').status_code,200)

    def test_incomplete_surface_coverage_cannot_replace_previous_index(self):
        first=self.build()
        units=voxelize(load_scene(str(self.model_path)),voxel_size=0.5)
        units[0].area *= 0.5
        with patch('index_service.voxelize',return_value=units):
            self.assertEqual(self.client.post('/reindex',json={'scene':'sample-room'}).status_code,202)
            job=self.wait_for_job()
            self.assertEqual(job['status'],'failed')
            self.assertIn('coverage validation failed',job['message'])
        self.assertEqual(self.manager.get(scene_fingerprint(self.model_path)).metadata['generation'],first['id'])

    def test_second_rebuild_is_rejected_while_running(self):
        release=threading.Event()
        original=load_scene
        def slow(path):
            release.wait(5)
            return original(path)
        with patch('index_service.load_scene',side_effect=slow):
            self.assertEqual(self.client.post('/reindex',json={'scene':'sample-room'}).status_code,202)
            try:self.assertEqual(self.client.post('/reindex',json={'scene':'sample-room'}).status_code,409)
            finally:release.set()
            self.assertEqual(self.wait_for_job()['status'],'complete')

    def test_rebuilding_another_scene_preserves_previous_index(self):
        first=self.build()
        other=self.root/'scenes'/'second-room'/'second.glb'
        write_glb(other,transformed=True)
        self.assertEqual(self.client.post('/reindex',json={'scene':'second-room'}).status_code,202)
        self.assertEqual(self.wait_for_job()['status'],'complete')
        self.assertEqual(self.manager.get(scene_fingerprint(self.model_path)).metadata['generation'],first['id'])
        self.assertTrue(all(entry['indexed'] for entry in self.client.get('/scenes').get_json()))
        restarted=IndexManager(self.root,FakeModel())
        self.assertIsNotNone(restarted.get(scene_fingerprint(self.model_path)))
        self.assertIsNotNone(restarted.get(scene_fingerprint(other)))

    def test_invalid_requests_and_unindexed_search(self):
        self.assertEqual(self.client.post('/reindex',json={'scene':'../missing'}).status_code,400)
        self.assertEqual(self.client.post('/reindex',json={'scene':'sample-room','voxel_size':0}).status_code,400)
        self.assertEqual(self.client.get('/search?q=wall&scene=sample-room').status_code,409)
        self.assertEqual(self.client.get('/search?q=wall&scene=missing').status_code,404)
        self.assertEqual(self.client.get('/search?q=').status_code,400)


if __name__=='__main__':unittest.main()
