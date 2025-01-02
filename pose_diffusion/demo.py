# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Optional, Union
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate, get_original_cwd
import visdom
import models
import time
from torch.utils.data import DataLoader
from datasets.camera import CameraParamsDataset
from functools import partial
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.ops import corresponding_cameras_alignment
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.vis.plotly_vis import plot_scene

from util.utils import seed_all_random_engines
from util.match_extraction import extract_match
from util.load_img_folder import load_and_preprocess_images
from util.geometry_guided_sampling import geometry_guided_sampling
from util.metric import compute_ARE
from visdom import Visdom


@hydra.main(config_path="../cfgs/", config_name="default")
def demo(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print("Model Config:")
    print(OmegaConf.to_yaml(cfg))

    # Check for GPU availability and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = instantiate(cfg.MODEL, _recursive_=False)

    # Load and preprocess images
    original_cwd = get_original_cwd()  # Get original working directory
    folder_path = os.path.join(original_cwd, cfg.image_folder)
    images, image_info = load_and_preprocess_images(folder_path, cfg.image_size)

    # Load camera parameters

    transforms_json_path = cfg.camera_dataloader.transforms_json
    camera_params_dataset = CameraParamsDataset(
        transforms_json_path=transforms_json_path,
    )
    camera_params_dataloader = DataLoader(
        camera_params_dataset,
        batch_size=cfg.camera_dataloader.camera_batch_size,
        shuffle=False,  
        num_workers=cfg.camera_dataloader.num_workers,
        pin_memory=cfg.camera_dataloader.pin_memory,
        persistent_workers=cfg.camera_dataloader.persistent_workers,
        drop_last=cfg.camera_dataloader.drop_last,
    )

    # Load checkpoint
    ckpt_path = os.path.join(original_cwd, cfg.ckpt)
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded checkpoint from: {ckpt_path}")
    else:
        raise ValueError(f"No checkpoint found at: {ckpt_path}")

    # Move model and images to the GPU
    model = model.to(device)
    images = images.to(device)

    # Evaluation Mode
    model.eval()

    # Seed random engines
    seed_all_random_engines(cfg.seed)

    # Start the timer
    start_time = time.time()

    camera_params_iter = iter(camera_params_dataloader)
    # Perform match extraction
    if cfg.GGS.enable:
        # Optional TODO: remove the keypoints outside the cropped region?

        kp1, kp2, i12 = extract_match(image_folder_path=folder_path, image_info=image_info)

        if kp1 is not None:
            keys = ["kp1", "kp2", "i12", "img_shape"]
            values = [kp1, kp2, i12, images.shape]
            matches_dict = dict(zip(keys, values))

            cfg.GGS.pose_encoding_type = cfg.MODEL.pose_encoding_type
            GGS_cfg = OmegaConf.to_container(cfg.GGS)

            cond_fn = partial(geometry_guided_sampling, matches_dict=matches_dict, GGS_cfg=GGS_cfg)
            print("[92m=====> Sampling with GGS <=====[0m")
        else:
            cond_fn = None
    else:
        cond_fn = None
        print("[92m=====> Sampling without GGS <=====[0m")

    images = images.unsqueeze(0)
    
    try:
        camera_params_batch = next(camera_params_iter)
    except StopIteration:
        raise ValueError("CameraParams DataLoader is exhausted. Ensure it has enough data.")
    # Move camera parameters to the GPU
    translations = camera_params_batch['T'].to(device)
    rotations = camera_params_batch['R'].to(device)
    fl = camera_params_batch['fl'].to(device)
    pp = camera_params_batch['pp'].to(device)

    spann3r_cameras = PerspectiveCameras(
        focal_length=fl.reshape(-1, 2)/100,
        R=rotations.reshape(-1, 3, 3),
        T=translations.reshape(-1, 3),
        device=device,
    )

    
    # Forward
    with torch.no_grad():
        # Obtain predicted camera parameters
        # pred_cameras is a PerspectiveCameras object with attributes
        # pred_cameras.R, pred_cameras.T, pred_cameras.focal_length

        # The poses and focal length are defined as
        # NDC coordinate system in
        # https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md
        predictions = model(image=images, cond_fn=cond_fn, cond_start_step=cfg.GGS.start_step, training=False, mid_camera=spann3r_cameras)

    pred_cameras = predictions["pred_cameras"]
    print(pred_cameras.T)
    # Stop the timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time taken: {:.4f} seconds".format(elapsed_time))

    # Compute metrics if gt is available

    # Load gt poses
    if os.path.exists(os.path.join(folder_path, "gt_cameras.npz")):
        gt_cameras_dict = np.load(os.path.join(folder_path, "gt_cameras.npz"))
        gt_cameras = PerspectiveCameras(
            focal_length=gt_cameras_dict["gtFL"], R=gt_cameras_dict["gtR"], T=gt_cameras_dict["gtT"], device=device
        )

        # 7dof alignment, using Umeyama's algorithm
        pred_cameras_aligned = corresponding_cameras_alignment(
            cameras_src=pred_cameras, cameras_tgt=gt_cameras, estimate_scale=True, mode="extrinsics", eps=1e-9
        )

        # Compute the absolute rotation error
        ARE = compute_ARE(pred_cameras_aligned.R, gt_cameras.R).mean()
        print(f"For {folder_path}: the absolute rotation error is {ARE:.6f} degrees.")
    else:
        print(f"No GT provided. No evaluation conducted.")


    # Visualization
    try:
        viz = visdom.Visdom(env='posediffusion')

        cams_show = {"ours_pred": pred_cameras, "ours_pred_aligned": pred_cameras_aligned, "gt_cameras": gt_cameras, "spann3r_cameras": spann3r_cameras}

        fig = plot_scene({f"{folder_path}": cams_show})

        viz.plotlyplot(fig, env="posediffusion", win="cams")
    except:
        print("Please check your visdom connection")



if __name__ == "__main__":
    demo()
