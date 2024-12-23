import numpy as np

# 读取 .npz 文件
npz_file = np.load('/home/mutian/Workspace/PoseDiffusion/pose_diffusion/samples/apple/gt_cameras.npz')

for key in npz_file.files:
    print(f"{key}: {npz_file[key]}")

