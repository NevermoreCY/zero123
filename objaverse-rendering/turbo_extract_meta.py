# first load the id , we will use name

import pandas as pd
import os
# Index(['index', 'ID', 'AssetName', 'Artist', 'Keywords', 'GeometryType',
#        'Polygons', 'Vertices', 'Textures', 'Materials', 'Rigged', 'Animated',
#        'UVMapped', 'UVMapType', 'Certification', 'Description', 'Filename',
#        'Format', 'FileVersion', 'Renderer', 'RendererVersion', 'RenderVersion',
#        'FeatureGraphNode', 'is_available', 'max', 'Feature Graph Node',
#        'Mesh State', 'Material State', 'Notes', 'Kit Version'],
#       dtype='object')

full = pd.read_pickle('turbosquid.p')
prefix = 'turbosquid/Commercial-Mammal-withTexture'
id_list = os.listdir(prefix)
# x = full.loc[full['ID'].isin(id_list)]
# x = full.loc[full['ID'] == 1420189]


c = 0
total = len(id_list)
for i in range(len(id_list)):
    c += 1
    print(c)
    idx = id_list[i]
    asset_names = full.loc[full['ID'] == int(idx)]
    asset_name = asset_names['AssetName'].iloc[0]
    out_file = prefix +'/' + idx + '/' + idx + '.txt'
    with open(out_file, 'w') as f:
        f.write(asset_name)
