#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ARCH_DIR = os.path.join(THIS_DIR, "architecture")
if ARCH_DIR not in sys.path:
    sys.path.insert(0, ARCH_DIR)

from encoder_model import load_brain_encoder, load_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone image->nsdgeneral encoder inference"
    )
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--subject", type=str, default="subj01", help="Subject id")
    parser.add_argument(
        "--subject_root",
        type=str,
        default=os.path.join(THIS_DIR, "model_zoo", "subjects"),
        help="Subject package root folder",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Override config path (default: <subject_root>/<subject>/config.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Override model checkpoint path (default: <subject_root>/<subject>/model.pth)",
    )
    parser.add_argument(
        "--coords",
        type=str,
        default="",
        help="Override coords path (default: <subject_root>/<subject>/coords.npy)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output npy path for predicted nsdgeneral signal",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--preserve_aspect_ratio",
        action="store_true",
        help="Use resize+center-crop instead of direct resize",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"image not found: {args.image}")

    subject_dir = os.path.join(args.subject_root, args.subject)
    config_path = args.config or os.path.join(subject_dir, "config.yaml")
    ckpt_path = args.checkpoint or os.path.join(subject_dir, "model.pth")
    coords_path = args.coords or os.path.join(subject_dir, "coords.npy")

    for p in [config_path, ckpt_path, coords_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"required file not found: {p}")

    print("=" * 70)
    print("Encoder inference")
    print("=" * 70)
    print(f"subject   : {args.subject}")
    print(f"image     : {args.image}")
    print(f"config    : {config_path}")
    print(f"checkpoint: {ckpt_path}")
    print(f"coords    : {coords_path}")
    print(f"device    : {args.device}")

    encoder = load_brain_encoder(
        subject=args.subject,
        config_path=config_path,
        checkpoint_path=ckpt_path,
        coords_path=coords_path,
        device=args.device,
    )

    image_tensor = load_image(
        image_path=args.image,
        image_resolution=tuple(encoder.cfg.DATASET.IMAGE_RESOLUTION),
        preserve_aspect_ratio=args.preserve_aspect_ratio,
    )

    with torch.no_grad():
        pred = encoder.predict_subject(image_tensor, subject=args.subject)

    pred = pred.detach().cpu().numpy().astype(np.float32)
    if pred.ndim == 2 and pred.shape[0] == 1:
        pred = pred[0]

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output, pred)

    print("-" * 70)
    print(f"saved: {args.output}")
    print(f"shape: {pred.shape}")
    print(f"range: [{pred.min():.6f}, {pred.max():.6f}]")
    print(f"mean : {pred.mean():.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
