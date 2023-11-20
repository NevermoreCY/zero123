import os

import cv2
import numpy as np
import json


views_folder = "views_shape"


folders = os.listdir(views_folder)
c = 0
total = len(folders)
print(total)
bad_path = []
good_path = []

good_dist = {}
bad_dist = {}

for folder in folders:
    img_path = views_folder + '/' + folder + '/000.png'
    img =cv2.imread(img_path)

    prompt_p = views_folder + '/' + folder + '/BLIP_best_text_v2.txt'
    with open(prompt_p,'r') as f:
        prompt = f.readline()

    if img.sum() == 328200:
        bad_path.append(folder)
        if prompt not in bad_dist:
            bad_dist[prompt] = [folder]
        else:
            bad_dist[prompt].append(folder)
        print(c , total ,folder)
        c+=1
    else:
        good_path.append(folder)
        if prompt not in good_dist:
            good_dist[prompt] = [folder]
        else:
            good_dist[prompt].append(folder)


out_path = 'shapenet_v3_good.json'
with open (out_path,'w') as f:
    json.dump(good_path,f)

out_path = 'shapenet_v3_good_dist.json'
with open (out_path,'w') as f:
    json.dump(good_dist,f)

out_path = 'shapenet_v3_bad.json'
with open (out_path,'w') as f:
    json.dump(bad_path,f)

out_path = 'shapenet_v3_bad_dist.json'
with open (out_path,'w') as f:
    json.dump(bad_dist,f)

for key in good_dist:
    print(key, len(good_dist[key]))

print(c, total, c/total)

# print("1071fa4cddb2da2fc8724d5673a063a6", "1071fa4cddb2da2fc8724d5673a063a6" in bad_path)


