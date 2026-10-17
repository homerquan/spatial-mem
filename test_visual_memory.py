"""RGB, visibility, depth, UV and pose regressions without model downloads."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from memory_render import ScanRenderer, unproject, camera_basis
from parse_glb import Scene, Triangle
from visual_memory import VisualMemory, pose, build_visual_memory
from voxelize import voxelize


def triangle(points, material=0, uv=None):
    points=np.asarray(points,dtype=float)
    normal=np.cross(points[1]-points[0],points[2]-points[0])
    normal/=np.linalg.norm(normal)
    return Triangle(*points,normal,'test','test','unknown',material,uv)


def material(rgb):
    return {'image':np.full((8,8,3),rgb,dtype=np.uint8), 'factor':[1.,1.,1.],
        'double_sided':False}


class VisualMemoryTests(unittest.TestCase):
    def test_depth_occlusion_and_pixel_registration_share_world_coordinates(self):
        triangles=[]
        for y,mat in [(2,0),(1,1)]:
            triangles.extend([triangle([[-2,y,-1],[2,y,-1],[-2,y,3]],mat),
                              triangle([[2,y,-1],[2,y,3],[-2,y,3]],mat)])
        scene=Scene(triangles=triangles,materials=[material([255,0,0]),material([0,0,255])])
        renderer=ScanRenderer(scene)
        eye=np.array([0.,0.,1.])
        rgb,depth,owners,basis=renderer.render(eye,[0,1,0],size=64)
        self.assertEqual(rgb[32,32].tolist(),[0,0,255])
        self.assertAlmostEqual(depth[32,32],1)
        self.assertIn(int(owners[32,32]),[2,3])
        points=unproject(depth,eye,basis,70)
        self.assertAlmostEqual(points[32,32,1],1)
        np.testing.assert_allclose(points[32,32],[.010940743,1,.989059257],atol=1e-8)

    def test_uv_texture_orientation_and_front_face_visibility(self):
        image=np.zeros((8,8,3),dtype=np.uint8)
        image[:4]=[255,0,0];image[4:]=[0,255,0]
        mat=material([0,0,0]);mat['image']=image
        tri=triangle([[-2,2,-1],[2,2,-1],[-2,2,3]],uv=np.array([[0,1],[1,1],[0,0]],dtype=float))
        renderer=ScanRenderer(Scene(triangles=[tri],materials=[mat]))
        rgb,_,owners,_=renderer.render([0,0,1],[0,1,0],size=64)
        self.assertEqual(rgb[10,10].tolist(),[255,0,0])
        self.assertEqual(rgb[52,10].tolist(),[0,255,0])
        _,_,back,_=renderer.render([0,4,1],[0,-1,0],size=64)
        self.assertTrue(np.all(back==-1),'the back of a one-sided scan is not an observation')

    def test_walkthrough_uses_supported_eye_level_positions_and_all_headings(self):
        triangles=[triangle([[-2,-2,0],[2,-2,0],[-2,2,0]]),
            triangle([[2,-2,0],[2,2,0],[-2,2,0]]),
            triangle([[-2,-2,3],[-2,2,3],[2,-2,3]]),
            triangle([[2,-2,3],[-2,2,3],[2,2,3]])]
        scene=Scene(triangles=triangles,bounds_min=np.array([-2.,-2.,0.]),bounds_max=np.array([2.,2.,3.]))
        path=ScanRenderer(scene).walkthrough()
        self.assertEqual(len(path),32)
        for eye,direction in path:
            self.assertAlmostEqual(eye[2],1.6)
            self.assertAlmostEqual(np.linalg.norm(direction),1)
        np.testing.assert_allclose(sum(direction[:2] for _,direction in path),[0,0],atol=1e-10)

    def test_pose_angles_and_camera_basis_follow_z_up(self):
        camera=pose([1,2,3],[0,1,1])
        self.assertAlmostEqual(camera['yaw'],90)
        self.assertAlmostEqual(camera['pitch'],45)
        basis=camera_basis(camera['direction'])
        np.testing.assert_allclose(basis@basis.T,np.eye(3),atol=1e-10)
        self.assertEqual(camera['up'],[0,0,1])

    def test_build_persists_images_depth_triangle_registration_and_camera_pose(self):
        scene=Scene(triangles=[triangle([[-2,2,-1],[2,2,-1],[-2,2,3]]),
            triangle([[2,2,-1],[2,2,3],[-2,2,3]])],
            bounds_min=np.array([-2.,2.,-1.]),bounds_max=np.array([2.,2.,3.]),
            materials=[material([100,150,200])])
        units=voxelize(scene,voxel_size=1)
        class Encoder:
            def encode(self,images=None,**_):return np.full((len(images),4),.5,dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ScanRenderer,'walkthrough',return_value=[(np.array([0.,0.,1.]),np.array([0.,1.,0.]))]):
                metadata=build_visual_memory(scene,units,directory,Encoder())
            memory=VisualMemory.read(directory)
            self.assertEqual(metadata['walkthrough_view_count'],1)
            self.assertEqual(len(memory.records),10)
            self.assertEqual(memory.embeddings.shape,(10,4))
            for record in memory.records:
                self.assertTrue((Path(directory)/'observations'/record['image']).exists())
                self.assertTrue(record['patch_ids'])
                self.assertEqual(record['camera']['position'],[0,0,1])
            data=np.load(Path(directory)/'frames'/'0000.npz',allow_pickle=False)
            self.assertEqual(data['depth'].shape,(256,256))
            self.assertAlmostEqual(data['depth'][128,128],2)
            self.assertGreaterEqual(data['triangle_ids'][128,128],0)
            self.assertGreaterEqual(data['patch_indices'][128,128],0)
            self.assertTrue((Path(directory)/'frames'/'poses.json').exists())

    def test_overlapping_crops_are_deduplicated_but_distinct_locations_remain(self):
        records=[]
        for index,(crop,center) in enumerate([([0,0,1,1],[0,0,0]),([0,0,.5,.5],[0,0,0]),([.5,.5,1,1],[5,5,0])]):
            records.append({'unit_id':str(index),'view_id':0 if index<2 else 1,
                'crop':crop,'center':center,'camera':pose([0,0,1],[1,0,0])})
        class Encoder:
            def encode(self,**_):return np.array([[1.,0.]])
        memory=VisualMemory(records,np.array([[1,0],[.9,.1],[.8,.2]]),Path('.'))
        self.assertEqual([r['unit_id'] for r in memory.search('object',Encoder(),3)],['0','2'])


if __name__=='__main__':unittest.main()
