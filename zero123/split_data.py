import os
import json
from tqdm import tqdm
import shutil


file_dir = "/yuch_ws/views_release"

root_dir = "/yuch_ws/zero123/zero123"

with open(os.path.join(root_dir, 'valid_paths.json')) as f:
    paths = json.load(f)


h = len(paths) // 4

fh = paths[:h]
sh = paths[h:2*h]
third = paths[2*h:3*h]
four = paths[3*h:]



fh_dir = "/yuch_ws/views_p1"
sh_dir = "/yuch_ws/views_p2"
fh_dir = "/yuch_ws/views_p3"
sh_dir = "/yuch_ws/views_p4"


cur = fh

for i in tqdm(range(len(cur))):
    original_path = file_dir + cur[i]
    target_path = fh_dir + cur[i]
    shutil.copy(original_path,target_path)



