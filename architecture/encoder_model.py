import os
from typing import Dict, Optional

import numpy as np
import torch
import torchvision.io as io
from torch import Tensor, nn

from config_utils import load_from_yaml
from models import DevMemVoxelWiseEncodingModel, MemVoxelWiseEncodingModel


def load_image(
    image_path: str,
    image_resolution=(224, 224),
    preserve_aspect_ratio: bool = False,
) -> Tensor:
    image = io.read_image(image_path).float() / 255.0

    if preserve_aspect_ratio:
        _, h, w = image.shape
        target_h, target_w = image_resolution
        scale = max(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        start_h = (new_h - target_h) // 2
        start_w = (new_w - target_w) // 2
        image = image[:, start_h : start_h + target_h, start_w : start_w + target_w]
    else:
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=image_resolution,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std
    return image.unsqueeze(0)


class BrainEncoder(nn.Module):
    """Minimal inference wrapper compatible with exported plmodel state_dict.

    This module keeps only the fields needed at inference time:
    - coord_dict
    - model
    """

    def __init__(self, cfg, coord_dict: Dict[str, np.ndarray]):
        super().__init__()
        self.cfg = cfg.clone()
        self.subject_list = list(coord_dict.keys())
        self.cfg.DATASET.SUBJECT_LIST = self.subject_list

        self.coord_dict = nn.ParameterDict(
            {
                subject: nn.Parameter(
                    torch.as_tensor(coords, dtype=torch.float32), requires_grad=False
                )
                for subject, coords in coord_dict.items()
            }
        )
        n_voxel_dict = {
            subject: int(self.coord_dict[subject].shape[0])
            for subject in self.subject_list
        }

        if self.cfg.EXPERIMENTAL.USE_DEV_MODEL:
            self.model = DevMemVoxelWiseEncodingModel(self.cfg, n_voxel_dict)
        else:
            self.model = MemVoxelWiseEncodingModel(self.cfg, n_voxel_dict)

    def _dummy_behavior(self, batch_size: int, device: torch.device):
        in_dim = int(self.cfg.MODEL.COND.IN_DIM)
        n_prev = int(self.cfg.DATASET.N_PREV_FRAMES)
        bhv = torch.zeros(batch_size, in_dim, device=device)
        prev_bhvs = torch.zeros(batch_size, n_prev + 1, in_dim, device=device)
        return bhv, prev_bhvs

    @torch.no_grad()
    def predict_subject(self, image_tensor: Tensor, subject: str) -> Tensor:
        if subject not in self.coord_dict:
            raise ValueError(f"Unknown subject: {subject}")

        image_tensor = image_tensor.to(next(self.parameters()).device)
        coords = self.coord_dict[subject]
        bsz = int(image_tensor.shape[0])
        bhv, prev_bhvs = self._dummy_behavior(bsz, image_tensor.device)

        prev_img: Optional[Tensor] = None
        prev_feats: Optional[Tensor] = None
        if getattr(self.cfg.EXPERIMENTAL, "USE_PREV_FRAME", False):
            prev_img = torch.zeros_like(image_tensor)
            prev_feats = torch.zeros(
                bsz,
                int(self.cfg.DATASET.N_PREV_FRAMES),
                int(self.cfg.MODEL.PREV_FEAT.DIM),
                device=image_tensor.device,
                dtype=image_tensor.dtype,
            )

        out = self.model(
            x=image_tensor,
            subject=subject,
            coords=coords,
            bhv=bhv,
            prev_img=prev_img,
            prev_feats=prev_feats,
            prev_bhvs=prev_bhvs,
            voxel_indices=None,
            chunk_size=int(self.cfg.MODEL.CHUNK_SIZE),
        )
        return out


def _normalize_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    has_model_prefix = any(k.startswith("model.") for k in state_dict.keys())
    has_coord_prefix = any(k.startswith("coord_dict.") for k in state_dict.keys())
    if has_model_prefix and has_coord_prefix:
        return state_dict

    fixed = {}
    for k, v in state_dict.items():
        if k.startswith("coord_dict.") or k.startswith("model."):
            fixed[k] = v
        elif k.startswith("module.coord_dict.") or k.startswith("module.model."):
            fixed[k.replace("module.", "", 1)] = v
        else:
            fixed[f"model.{k}"] = v
    return fixed


def load_brain_encoder(
    subject: str,
    config_path: str,
    checkpoint_path: str,
    coords_path: str,
    device: str = "cpu",
) -> BrainEncoder:
    cfg = load_from_yaml(config_path)
    cfg.DATASET.SUBJECT_LIST = [subject]

    coords = np.load(coords_path)
    encoder = BrainEncoder(cfg=cfg, coord_dict={subject: coords})

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = _normalize_state_dict(state_dict)

    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[warn] missing keys: {len(missing)}")

    encoder.to(device)
    encoder.eval()
    return encoder
