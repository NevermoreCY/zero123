# import os
# from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Image, Texture, Material
# from pygltflib.utils import GLTFConverter
#
# def obj_to_glb(input_obj_file, output_glb_file):
#     # Step 1: Convert OBJ to GLTF
#     temp_gltf_file = "temp.gltf"
#     GLTFConverter(obj_file=input_obj_file, output_gltf_file=temp_gltf_file).convert()
#
#     # Step 2: Load the temporary GLTF file
#     gltf = GLTF2().load(temp_gltf_file)
#
#     # Step 3: Save as GLB
#     gltf.save_binary(output_glb_file)
#
#     # Remove the temporary GLTF file
#     os.remove(temp_gltf_file)
#     print(f"Conversion completed: '{input_obj_file}' to '{output_glb_file}'")
#
# # Example usage
# input_obj_file = "test/267296.obj"  # Update with the path to your OBJ file
# output_glb_file = "test/267296.glb"  # Update with your desired output path
# obj_to_glb(input_obj_file, output_glb_file)


import bpy
import sys

# Paths
input_obj_path = sys.argv[5]  # Path to the input OBJ file
output_glb_path = sys.argv[6] # Path to the output GLB file

# Delete default objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import OBJ file
bpy.ops.import_scene.obj(filepath=input_obj_path)

# Export to GLB
bpy.ops.export_scene.gltf(filepath=output_glb_path, export_format='GLB')

print("Conversion completed: OBJ to GLB")

# blender-3.2.2-linux-x64/blender --background --python obj_to_glb.py -- shapenet/cat/371028.obj shapenet/cat/371028.glb