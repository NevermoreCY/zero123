import os
import json
from tqdm import tqdm
import shutil

from distutils.dir_util import copy_tree


file_dir = "/yuch_ws/views_release"

root_dir = "/yuch_ws/zero123/zero123"

with open(os.path.join(root_dir, 'valid_paths.json')) as f:
    paths = json.load(f)


h = len(paths) // 4

one = paths[:h]
two = paths[h:2*h]
third = paths[2*h:3*h]
four = paths[3*h:]



one_dir = "/yuch_ws/views_p1"
two_dir = "/yuch_ws/views_p2"
third_dir = "/yuch_ws/views_p3"
four_dir = "/yuch_ws/views_p4"


cur = four
cur_dir = four_dir
print("*** current dir is ", cur_dir)
for i in tqdm(range(len(cur))):
    original_path = file_dir + "/" + cur[i]
    target_path = cur_dir + "/" + cur[i]
    copy_tree(original_path,target_path)



