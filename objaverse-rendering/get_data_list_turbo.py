import json
import os

view_dir = 'views_turbo'
data_list = os.listdir(view_dir)
out_path = 'turbo_v1.json'
print("num of samples: " , len(data_list))
with open(out_path, 'w') as f:
    json.dump(data_list,f)