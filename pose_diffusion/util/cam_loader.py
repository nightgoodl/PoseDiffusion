import os
from torch.utils.data import DataLoader
from datasets import CameraParamsDataset
from omegaconf import DictConfig

def get_camera_params_dataloader(cfg: DictConfig) -> DataLoader:

    transforms_json_path = cfg.train.transforms_json

    camera_params_dataset = CameraParamsDataset(transforms_json_path=transforms_json_path)

    batch_size = cfg.train.camera_batch_size

    dataloader = DataLoader(
        camera_params_dataset,
        batch_size=batch_size,
        shuffle=False,                     
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
        drop_last=True,                   
    )
    #test

    return dataloader
