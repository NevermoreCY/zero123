import os

prefix = 'logs/2023-11-07T18-23-24_zero123_shape_turbo/checkpoints'

ckpts = os.listdir(prefix)

for ckpt in ckpts:
    epoch_n = (ckpt.split('-')[0]).split('=')[1]
    print(epoch_n)
