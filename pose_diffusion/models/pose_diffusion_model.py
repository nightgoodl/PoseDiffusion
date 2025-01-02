# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Standard library imports
import base64
import io
import logging
import math
import pickle
import warnings
from collections import defaultdict
from dataclasses import field, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party library imports
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.transforms import se3_exp_map, se3_log_map, Transform3d, so3_relative_angle
from util.camera_transform import pose_encoding_to_camera, camera_to_pose_encoding

import models
from hydra.utils import instantiate
from pytorch3d.renderer.cameras import PerspectiveCameras

logger = logging.getLogger(__name__)


class PoseDiffusionModel(nn.Module):
    def __init__(self, pose_encoding_type: str, IMAGE_FEATURE_EXTRACTOR: Dict, DIFFUSER: Dict, DENOISER: Dict):
        """Initializes a PoseDiffusion model.

        Args:
            pose_encoding_type (str):
                Defines the encoding type for extrinsics and intrinsics
                Currently, only `"absT_quaR_logFL"` is supported -
                a concatenation of the translation vector,
                rotation quaternion, and logarithm of focal length.
            image_feature_extractor_cfg (Dict):
                Configuration for the image feature extractor.
            diffuser_cfg (Dict):
                Configuration for the diffuser.
            denoiser_cfg (Dict):
                Configuration for the denoiser.
        """

        super().__init__()

        self.pose_encoding_type = pose_encoding_type

        self.image_feature_extractor = instantiate(IMAGE_FEATURE_EXTRACTOR, _recursive_=False)
        self.diffuser = instantiate(DIFFUSER, _recursive_=False)

        denoiser = instantiate(DENOISER, _recursive_=False)
        self.diffuser.model = denoiser

        self.target_dim = denoiser.target_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        image: torch.Tensor,
        gt_cameras: Optional[CamerasBase] = None,
        mid_camera: Optional[CamerasBase] = None,  # 中间相机参数
        sequence_name: Optional[List[str]] = None,
        cond_fn=None,
        cond_start_step=0,
        training=True,
        batch_repeat=-1,
    ):
        """
        Forward pass of the PoseDiffusionModel.

        Args:
            image (torch.Tensor):
                Input image tensor, Bx3xHxW.
            gt_cameras (Optional[CamerasBase], optional):
                Camera object. Defaults to None.
            sequence_name (Optional[List[str]], optional):
                List of sequence names. Defaults to None.
            cond_fn ([type], optional):
                Conditional function. Wrapper for GGS or other functions.
            cond_start_step (int, optional):
                The sampling step to start using conditional function.

        Returns:
            PerspectiveCameras: PyTorch3D camera object.
        """

        shapelist = list(image.shape)
        batch_num = shapelist[0]
        frame_num = shapelist[1]

        reshaped_image = image.reshape(batch_num * frame_num, *shapelist[2:])
        z = self.image_feature_extractor(reshaped_image).reshape(batch_num, frame_num, -1)
        if training:
            pose_encoding = camera_to_pose_encoding(gt_cameras, pose_encoding_type=self.pose_encoding_type)

            if batch_repeat > 0:
                pose_encoding = pose_encoding.reshape(batch_num * batch_repeat, -1, self.target_dim)
                z = z.repeat(batch_repeat, 1, 1)
            else:
                pose_encoding = pose_encoding.reshape(batch_num, -1, self.target_dim)

            diffusion_results = self.diffuser(pose_encoding, z=z)

            diffusion_results["pred_cameras"] = pose_encoding_to_camera(
                diffusion_results["x_0_pred"], pose_encoding_type=self.pose_encoding_type
            )

            return diffusion_results
        else:
            if mid_camera is None:
                raise ValueError("mid_camera must be provided during inference.")
            
            #encode the mid camera
            pose_encoding_mid = camera_to_pose_encoding(mid_camera, pose_encoding_type=self.pose_encoding_type)
            if batch_repeat > 0:
                pose_encoding_mid = pose_encoding_mid.reshape(batch_num * batch_repeat, -1, self.target_dim)
                z = z.repeat(batch_repeat, 1, 1)
            else:
                pose_encoding_mid = pose_encoding_mid.reshape(batch_num, -1, self.target_dim)

            if pose_encoding_mid.shape[1] != z.shape[1]:
                pose_encoding_mid = pose_encoding_mid.repeat(1, z.shape[1], 1)
            
            # Add noise to pose_encoding_mid
            # Randomly sample timesteps for each instance in the batch
            """ num_noise_steps = int(self.diffuser.num_timesteps // 10)
            x = pose_encoding_mid

            # Randomly sample timesteps for each instance in the batch
            start_timestep = self.diffuser.num_timesteps - num_noise_steps  
            timesteps = torch.linspace(start_timestep, self.diffuser.num_timesteps - 1, num_noise_steps, device=x.device).long()

            # keep the dimension
            if len(timesteps) == 0:
                raise ValueError(f"Generated empty timesteps sequence. num_timesteps={self.diffuser.num_timesteps}, num_noise_steps={num_noise_steps}")

            for noise_step in range(num_noise_steps):
                t = timesteps[noise_step:noise_step+1]  # keep the dimension

                # generate noise
                noise = torch.randn_like(x)

                # add noise
                x = self.diffuser.q_sample(x_start=x, t=t, noise=noise)
                print(f"noise add step {noise_step+1}/{num_noise_steps}: t = {t.item()}, x shape {x.shape}")

 """
            noise = torch.randn_like(pose_encoding_mid).to(pose_encoding_mid.device)
            # Iterate over timesteps in reverse
            for _i,step in enumerate(reversed(range(self.diffuser.num_timesteps))):
                if _i < 30:
                    x = self.diffuser.q_sample(x_start=pose_encoding_mid, \
                                               t=torch.tensor(step).unsqueeze(-1).to(noise.device), noise=noise)
                else:
                    x, _ = self.diffuser.p_sample(
                        x=x,
                        t=step,
                        z=z,
                        cond_fn=cond_fn,
                        cond_start_step=cond_start_step
                    )
            # Convert the denoised pose encoding to cameras
            pred_cameras = pose_encoding_to_camera(x, pose_encoding_type=self.pose_encoding_type)

            diffusion_results = {"pred_cameras": pred_cameras, "z": z}

            return diffusion_results
