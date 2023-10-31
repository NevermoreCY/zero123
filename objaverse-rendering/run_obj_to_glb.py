# Importing required module
import subprocess
import os
# Using system() method to
# execute shell commands

id_list= os.listdir('turbosquid/Commercial-Mammal-withTexture')

for id in id_list:

    target_obj = 'turbosquid/Commercial-Mammal-withTexture' + '/' + id + '/' + id + '.obj'
    target_glb = ' turbosquid/Commercial-Mammal-withTexture' + '/' + id + '/' + id + '.glb'
    cmd = 'blender-3.2.2-linux-x64/blender --background --python obj_to_glb.py -- '+ target_obj+  target_glb
    print('runing ', cmd)
    subprocess.Popen(cmd, shell=True)