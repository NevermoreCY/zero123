"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --job_num my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
# import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
import os
import bpy
from mathutils import Vector
# from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--job_num",
    type=int,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="~/.objaverse/hf-objaverse-v1/views_whole_sphere")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=8)
parser.add_argument("--camera_dist", type=float, default=1.2)


argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# setup lighting
bpy.ops.object.light_add(type='POINT',radius=10)
print('keys: ', bpy.data.lights.keys())
light2 = bpy.data.lights['Point']
# # bpy.ops.object.light_add(radius=10, location=(2,0,0), scale=(100,100,100))
# # bpy.ops.object.light_add(radius=10,location=(-2,0,0), scale=(100,100,100))
# # bpy.ops.object.light_add(radius=10,location=(0,2,0), scale=(100,100,100))
# # bpy.ops.object.light_add(radius=10,location=(0,-2,0), scale=(100,100,100))
# # bpy.ops.object.light_add(radius=10,location=(0,0,-2), scale=(100,100,100))
# # bpy.ops.object.light_add(radius=10,location=(0,0,2), scale=(100,100,100))
# light2 = bpy.data.lights["Point"]
light2.energy = 1000
bpy.data.objects['Point'].location[2] = -0.5
bpy.data.objects['Point'].scale[0] = 1000
bpy.data.objects['Point'].scale[1] = 1000
bpy.data.objects['Point'].scale[2] = 1000

import math
# Function to add an area light at a specific location and rotation
# def add_area_light(name, location, rotation_euler):
#     bpy.ops.object.light_add(type='AREA', location=location)
#     light = bpy.context.object
#     light.name = name
#     light.data.shape = 'RECTANGLE'
#     light.data.size = 2.0  # Size of the light; adjust as needed
#     light.rotation_euler = rotation_euler
#     return light

# Define locations and rotations for 6 lights
# light_positions_rotations = [
#     (("AreaLight1", (1, 1, 1), (math.radians(45), 0, 0))),
#     (("AreaLight2", (-1, -1, 1), (math.radians(-45), 0, 0))),
#     (("AreaLight3", (1, -1, 1), (0, math.radians(45), 0))),
#     (("AreaLight4", (-1, 1, 1), (0, math.radians(-45), 0))),
#     (("AreaLight5", (0, 0, ), (math.radians(-90), 0, 0))),
#     (("AreaLight6", (0, 0, -1), (math.radians(90), 0, 0)))
# ]

# # Add lights to the scene
# for name, location, rotation in light_positions_rotations:
#     add_area_light(name, location, rotation)
#
# print("All area lights added and configured.")

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"

def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

def sample_spherical(radius=3.0, maxz=3.0, minz=0.):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
#         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def randomize_camera():
    elevation = random.uniform(0., 90.)
    azimuth = random.uniform(0., 360)
    distance = random.uniform(0.8, 1.6)
    return set_camera_location(elevation, azimuth, distance)

def set_camera_location(elevation, azimuth, distance):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    # now light is the same as view point
    light2.energy = random.uniform(400, 800)
    bpy.data.objects["Point"].location[0] = x
    bpy.data.objects["Point"].location[1] = y
    bpy.data.objects["Point"].location[2] = z



    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    # light2.rotation_euler = rot_quat.to_euler()
    return camera

def randomize_lighting() -> None:
    light2.energy = random.uniform(500, 600)
    bpy.data.objects["Area"].location[0] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[1] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)


def reset_lighting() -> None:
    light2.energy = 1000
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 0
    bpy.data.objects["Area"].location[2] = 0.5


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        print('object_path :' , object_path)
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")





def save_images(object_file: str , name) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)

    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]

    # bpy.ops.object.metaball_add(type='BALL', radius=0.05, align='WORLD', location=(0.0, 0.0, -0.5),
    #                             rotation=(0.0, 0.0, 0.0), scale=(1, 1, 1))

    # bpy.ops.object.light_add(radius=10, location=(2,0,0), scale=(100,100,100))
    # bpy.ops.object.light_add(radius=10,location=(-2,0,0), scale=(100,100,100))
    # bpy.ops.object.light_add(radius=10,location=(0,2,0), scale=(100,100,100))
    # bpy.ops.object.light_add(radius=10,location=(0,-2,0), scale=(100,100,100))
    # bpy.ops.object.light_add(type='AREA',location=(0,0,-0.5), scale=(100,100,100))
    # bpy.ops.object.light_add(radius=10,location=(0,0,2), scale=(100,100,100))


    normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # randomize_lighting()
    for i in range(args.num_images):
        # # set the camera position
        # theta = (i / args.num_images) * math.pi * 2
        # phi = math.radians(60)
        # point = (
        #     args.camera_dist * math.sin(phi) * math.cos(theta),
        #     args.camera_dist * math.sin(phi) * math.sin(theta),
        #     args.camera_dist * math.cos(phi),
        # )
        # # reset_lighting()
        # cam.location = point

        # set camera
        camera = randomize_camera()

        # render the image
        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        # save camera RT matrix
        RT = get_3x4_RT_matrix_from_blender(camera)
        RT_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.npy")
        np.save(RT_path, RT)

        # save prompt
        prompt = name
        text_path = os.path.join(args.output_dir, object_uid, 'BLIP_best_text_v2.txt')
        with open(text_path,'w') as f:
            f.write(prompt)


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":

    # first load the id , we will use name



    # Index(['index', 'ID', 'AssetName', 'Artist', 'Keywords', 'GeometryType',
    #        'Polygons', 'Vertices', 'Textures', 'Materials', 'Rigged', 'Animated',
    #        'UVMapped', 'UVMapType', 'Certification', 'Description', 'Filename',
    #        'Format', 'FileVersion', 'Renderer', 'RendererVersion', 'RenderVersion',
    #        'FeatureGraphNode', 'is_available', 'max', 'Feature Graph Node',
    #        'Mesh State', 'Material State', 'Notes', 'Kit Version'],
    #       dtype='object')

    # full = pd.read_pickle('turbosquid.p')
    prefix = 'turbosquid/Commercial-Mammal-withTexture'
    id_list = os.listdir(prefix)
    # x = full.loc[full['ID'].isin(id_list)]
    # x = full.loc[full['ID'] == 1420189]
    job_num = args.job_num

    c = 0
    total = len(id_list)
    for i in range(len(id_list)):
        stat_t = time.time()
        c += 1
        idx = id_list[i]
        # asset_names = full.loc[full['ID'] == int(idx)]
        with open(prefix + '/'+ idx + '/' + idx + '.txt' , 'r') as f:
            name = f.readline()
        name = name.strip()

        print('*** processing file : ', prefix + '/'+ idx + '/' + idx + '.glb')
        if os.path.exists(args.output_dir + '/'+ idx  + '/011.png'):
            print('&&& skip file : ', args.output_dir + '/'+ idx  + '/011.png')
            continue
        # print('old path ', local_path)
        local_path = prefix + '/'+ idx + '/' + idx +'.glb'

        print('path ', local_path)
        save_images(local_path, name)
        end_i = time.time()
        time_cost = end_i - stat_t
        print("count: ", c, total, c / total, 'time cost : ', time_cost, 'time remaining(min) : ',
              ((total - c) * time_cost) / 60)
