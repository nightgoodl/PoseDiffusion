import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

class CameraParamsDataset(Dataset):
    def __init__(
        self,
        transforms_json_path: str,
        transform: Optional[Any] = None,
    ):
        with open(transforms_json_path, 'r') as f:
            self.data = json.load(f)
        
        self.frames = self.data['frames']
        self.transform = transform

        self.file_path_to_idx = {
            os.path.normpath(frame['file_path']): idx for idx, frame in enumerate(self.frames)
        }

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        frame = self.frames[idx]
        file_path = os.path.normpath(frame['file_path']) 

        # extract intrinsics
        intrinsics = frame['intrinsics']
        fl_x = intrinsics['fl_x']
        fl_y = intrinsics['fl_y']
        cx = intrinsics['cx']
        cy = intrinsics['cy']
        # extract distortion coefficients
        # k1, k2, p1, p2 = intrinsics['k1'], intrinsics['k2'], intrinsics['p1'], intrinsics['p2']

        # extract extrinsics
        transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        rotation_matrix = transform_matrix[:3, :3]
        translation = transform_matrix[:3, 3]

        # focal length and principal point
        fl = torch.tensor([fl_x, fl_y], dtype=torch.float32)
        pp = torch.tensor([cx, cy], dtype=torch.float32)

        sample = {
            'file_path': file_path,  
            'R': rotation_matrix,
            'T': translation,
            'fl': fl,
            'pp': pp,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
