import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const THREE = await import(process.env.THREE_MODULE || 'three');
const html = fs.readFileSync(new URL('./viewer.html', import.meta.url), 'utf8');
let script = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
script = script.replace(/^import .*;$/gm, '');
script = script.slice(0, script.lastIndexOf('init().catch'));
const elements = new Map();
const events = new Map();
const context2d = new Proxy({}, {get: (_, key) => key === 'canvas' ? {} : () => {}});
const element = id => {
  if (!elements.has(id)) elements.set(id, {textContent: '', disabled: false, style: {}, children: [], innerHTML: '',
    setAttribute(){}, focus(){}, addEventListener(name, fn){events.set(id+':'+name, fn);},
    getContext(){return context2d;}});
  return elements.get(id);
};
const windowMock = {location:{origin:'http://localhost'}, addEventListener(name, fn){events.set('window:'+name, fn);}};
const documentMock = {getElementById:element, createElement:tag=>({...element('new-'+tag)}),
  addEventListener(name, fn){events.set('document:'+name, fn);}, pointerLockElement:null};
const ctx = vm.createContext({THREE, window:windowMock, document:documentMock, console:{error(){}}, requestAnimationFrame(){}, fetch(){throw Error('unexpected fetch');}});
vm.runInContext(script,ctx);
const run = code=>vm.runInContext(code, ctx);
run(`
scene3d = new THREE.Scene();
camera = new THREE.PerspectiveCamera(); camera.up.set(0,0,1); camera.position.set(2,2,2);
controls = {enabled:true,target:new THREE.Vector3(), update(){},maxDistance:500};
renderer = {domElement:$('canvas')};
statusEl=$('status'); resultsEl=$('results'); searchBtn=$('searchBtn'); sceneSel=$('scene');
glbScene = new THREE.Group();
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.Float32BufferAttribute([0,.1,0,.9,.1,0,0,.9,0,3,3,0],3));
geometry.setIndex([0,1,2,1,3,2]);
const sourceMaterial = new THREE.MeshBasicMaterial();
const sourceMesh = new THREE.Mesh(geometry, sourceMaterial);
sourceMesh.position.set(10,-4,2); glbScene.add(sourceMesh); scene3d.add(glbScene);
matchedUnits.push({bbox_min:[9.9,-4.1,1.9],bbox_max:[11,-3,3],center:[10.3,-3.6,2],color:new THREE.Color(0xff0000)});
applyHeatmap();
`);
assert.equal(run('heatmapObjects.length'),1);
assert.ok(run('heatmapObjects[0].geometry.attributes.position.count')>3, 'crossing triangles are clipped even when their centroid is outside');
const overlayPosition = run('heatmapObjects[0].geometry.attributes.position');
for (let i=0;i<overlayPosition.count;i++) {
  assert.ok(overlayPosition.getX(i)>=9.9-1e-6 && overlayPosition.getX(i)<=11+1e-6);
  assert.ok(overlayPosition.getY(i)>=-4.1-1e-6 && overlayPosition.getY(i)<=-3+1e-6);
}
run(`const clipped = clipTriangleToBox([new THREE.Vector3(-2,0,0),new THREE.Vector3(2,0,0),new THREE.Vector3(0,2,0)],new THREE.Vector3(0,0,0),new THREE.Vector3(1,1,1));`);
assert.ok(run('clipped.length')>0);
assert.ok(Math.abs(run('clipped.reduce((sum,tri)=>sum+new THREE.Vector3().subVectors(tri[1],tri[0]).cross(new THREE.Vector3().subVectors(tri[2],tri[0])).length()/2,0)')-1)<1e-9);

assert.equal(run('sourceMesh.geometry === geometry'),true, 'original geometry is preserved');
assert.equal(run('sourceMaterial.vertexColors'),false, 'original material is preserved');
assert.equal(run('heatmapObjects[0].material.opacity'),.55);
assert.equal(run('heatmapObjects[0].geometry.attributes.color.itemSize'),4);
assert.ok(run('heatmapObjects[0].geometry.attributes.position.getX(0)') >=10, 'overlay uses world transforms');
run(`addHighlightBox({bbox_min:[0,0,0],bbox_max:[1,1,1],center:[.1,.2,.3]},1,1);`);
assert.deepEqual(Array.from(run('highlightObjects[0].position.toArray()')),[.5,.5,.5]);
assert.equal(run('highlightObjects[0].visible'),false);
assert.deepEqual(Array.from(run('highlightObjects[0].box.min.toArray()')),[0,0,0]);
assert.deepEqual(Array.from(run('highlightObjects[0].box.max.toArray()')),[1,1,1]);
run('clearHighlights();');
assert.equal(run('heatmapObjects.length'),0);
assert.equal(run('highlightObjects.length'),0);
run('camera.position.set(0,0,1.65); walkYaw=0; walkPitch=0; walkKeys.add("KeyW"); updateWalk(.5);');
assert.deepEqual(Array.from(run('camera.position.toArray()')),[1,0,1.65], 'forward respects Z-up');
run('walkKeys.clear();walkKeys.add("KeyE");updateWalk(.5);');
assert.equal(run('camera.position.z'),2.65);
run('walkKeys.clear();walkKeys.add("KeyW");walkKeys.add("KeyD");updateWalk(.5);');
assert.ok(Math.abs(run('camera.position.distanceTo(new THREE.Vector3(1,0,2.65))')-1)<1e-9, 'diagonal speed stays normalized');
run('setupWalkControls();walkMode=true;walkKeys.clear();');
events.get('window:keydown')({code:'KeyW',target:{closest:()=>({})},preventDefault(){throw Error('typing swallowed');}});
assert.equal(run('walkKeys.size'),0, 'typing cannot move camera');
events.get('window:keydown')({code:'KeyW',target:{closest:()=>null},preventDefault(){}});
assert.equal(run('walkKeys.has("KeyW")'),true);
events.get('window:blur')();
assert.equal(run('walkKeys.size'),0);
run(`loader={loadAsync:()=>Promise.reject(new Error('missing model'))};sceneFiles.set('missing',{glb_url:'/missing.glb'});`);
await run('loadScene("missing")');
assert.match(elements.get('status').textContent,/Failed to load missing: missing model/);
run(`
const deferred = new Map();
loader={loadAsync:url=>new Promise(resolve=>deferred.set(url,resolve))};
sceneFiles.set('first',{glb_url:'/first.glb',indexed:false});
sceneFiles.set('second',{glb_url:'/second.glb',indexed:false});
const firstLoad=loadScene('first'); const secondLoad=loadScene('second');
const secondScene=new THREE.Group();secondScene.add(new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshBasicMaterial()));
deferred.get('/second.glb')({scene:secondScene});
`);
await run('secondLoad');
run(`const firstScene = new THREE.Group(); deferred.get('/first.glb')({scene:firstScene});`);
await run('firstLoad');
assert.equal(run('glbScene===secondScene'),true,'late loads cannot replace selected room');
assert.match(elements.get('status').textContent,/Loaded second/);
run(`
walkMode=false; resultsEl.children=[];
glbScene=new THREE.Group();
const focusWall=new THREE.Mesh(new THREE.PlaneGeometry(4,4),new THREE.MeshBasicMaterial());
focusWall.rotation.x=Math.PI/2; focusWall.position.z=1.5; glbScene.add(focusWall);
camera.position.set(0,2,1.5);
const focusUnit={center:[0,0,1.5],bbox_min:[-.25,0,1.25],bbox_max:[.25,0,1.75],normal:[0,1,0]};
const visibleFocus=surfaceFocusPose(focusUnit);
`);
assert.ok(run('visibleFocus.position.y')<0,'a reversed index normal cannot send the camera behind a one-sided wall');
run(`
const blocker=new THREE.Mesh(new THREE.PlaneGeometry(.3,3),new THREE.MeshBasicMaterial({side:THREE.DoubleSide}));
blocker.rotation.x=Math.PI/2; blocker.position.set(0,-.4,1.5); glbScene.add(blocker);
const clearFocus=surfaceFocusPose(focusUnit);
const focusRay=new THREE.Raycaster(clearFocus.position,clearFocus.target.clone().sub(clearFocus.position).normalize());
`);
assert.equal(run('focusRay.intersectObject(glbScene,true)[0].object===focusWall'),true,'focus candidates must have an unobstructed view of the requested patch');
run(`
controls.enableDamping=true;
const savedFocus={...focusUnit,camera:{position:[2,-3,1.6],direction:[0,1,0],target:[2,0,1.6],fov:38,yaw:90,pitch:0}};
camera.aspect=1.5;
flyTo(savedFocus,1,1);
`);
assert.deepEqual(Array.from(run('camera.position.toArray()')),[2,-3,1.6]);
assert.ok(run('camera.getWorldDirection(new THREE.Vector3()).distanceTo(new THREE.Vector3(0,1,0))')<1e-10,'retrieval restores the recorded view angle');
assert.equal(run('camera.fov'),38);
assert.equal(run('controls.enableDamping'),true);
assert.match(elements.get('viewPose').textContent,/YAW 90.0° · PITCH 0.0°/);
run(`walkMode=true;flyTo(savedFocus,1,1);updateWalk(0);`);
assert.ok(run('camera.getWorldDirection(new THREE.Vector3()).distanceTo(new THREE.Vector3(0,1,0))')<1e-10,'walking continues from the saved orientation');
console.log('PASS: surface clipping, world bounds, Z-up walking, load races, reversed normals, occluded focus candidates, saved camera pose/FOV, and walk orientation after retrieval.');
