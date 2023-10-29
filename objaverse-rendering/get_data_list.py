import json
import os

view_dir = 'views'
data_list = os.listdir(view_dir)
out_path = 'shapenet_v1.json'
print("num of samples: " , data_list)
with open(out_path, 'w') as f:
    json.dump(data_list,f)