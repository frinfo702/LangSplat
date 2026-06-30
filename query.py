#!/usr/bin/env python
"""
Query script for LangSplat: text-based search on rendered language features.

Usage:
    cd LangSplat
    python query.py \
        --dataset_name sofa \
        --feat_dir output \
        --ae_ckpt_dir autoencoder/ckpt \
        --output_dir query_result \
        --image_dir output/sofa_3/train/ours_None/renders \
        --query "grey sofa"
"""
from __future__ import annotations

import os
import glob
import sys
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add eval to path for colormaps internal imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval"))

from autoencoder.model import Autoencoder
from eval.openclip_encoder import OpenCLIPNetwork
from eval import colormaps


def query_features(
    feat_dir: list[str],
    output_path: str,
    ae_ckpt_path: str,
    query_texts: list[str],
    image_dir: str | None = None,
    mask_thresh: float = 0.4,
    encoder_dims: list[int] | None = None,
    decoder_dims: list[int] | None = None,
):
    """Run text queries against rendered LangSplat language features."""
    if encoder_dims is None:
        encoder_dims = [256, 128, 64, 32, 3]
    if decoder_dims is None:
        decoder_dims = [16, 32, 64, 128, 256, 256, 512]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    # Count available feature files
    feat_paths_lvl0 = sorted(
        glob.glob(os.path.join(feat_dir[0], "*.npy")),
        key=lambda f: int(os.path.basename(f).split(".npy")[0]),
    )
    n_views = len(feat_paths_lvl0)
    print(f"Found {n_views} views, {len(feat_dir)} feature levels")

    # Determine image dimensions from first feature
    sample = np.load(feat_paths_lvl0[0])
    h, w = sample.shape[:2]

    # Load all compressed features (3 levels x N views x H x W x 3)
    compressed_sem_feats = np.zeros(
        (len(feat_dir), n_views, h, w, 3), dtype=np.float32
    )
    for i, fd in enumerate(feat_dir):
        feat_paths = sorted(
            glob.glob(os.path.join(fd, "*.npy")),
            key=lambda f: int(os.path.basename(f).split(".npy")[0]),
        )
        for j in range(n_views):
            compressed_sem_feats[i][j] = np.load(feat_paths[j])

    # Load autoencoder
    print("Loading autoencoder...")
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_dims, decoder_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Load CLIP
    print("Loading CLIP model...")
    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives(query_texts)

    # Load RGB images if available
    rgb_images = None
    if image_dir is not None:
        rgb_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not rgb_paths:
            rgb_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        if rgb_paths:
            rgb_images = []
            for p in rgb_paths[:n_views]:
                img = cv2.imread(p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                rgb_images.append(img)
            print(f"Loaded {len(rgb_images)} RGB images")


    # Process each view
    for idx in tqdm(range(n_views), desc="Querying"):
        # Load compressed features for this view across all levels
        sem_feat = torch.from_numpy(compressed_sem_feats[:, idx]).float().to(device)
        # sem_feat: (lvl, H, W, 3)

        lvl, h_, w_, c = sem_feat.shape

        # Decode from 3D -> 512D
        restored_feat = model.decode(sem_feat.flatten(0, 2))
        restored_feat = restored_feat.view(lvl, h_, w_, -1)  # lvl x H x W x 512

        # Get CLIP relevance map across all levels and prompts
        relev_map = clip_model.get_max_across(restored_feat)  # lvl x n_prompts x H x W
        n_head, n_prompt, _, _ = relev_map.shape

        # Get RGB image for this view
        rgb_img = None
        if rgb_images is not None and idx < len(rgb_images):
            rgb_img = torch.from_numpy(rgb_images[idx]).float().to(device) / 255.0

        for k, query_text in enumerate(query_texts):
            # For each prompt, find the best level
            score_lvl = torch.zeros(n_head, device=device)
            for i in range(n_head):
                score_lvl[i] = relev_map[i, k].max()
            chosen_lvl = torch.argmax(score_lvl).item()

            # Create heatmap
            heatmap = relev_map[chosen_lvl, k].unsqueeze(-1)  # H x W x 1
            heatmap_img = colormaps.apply_colormap(
                heatmap / (heatmap.max() + 1e-6),
                colormap_options,
            ).cpu().numpy()

            # Save heatmap
            safe_query = query_text.replace(" ", "_")
            save_dir = Path(output_path) / f"view_{idx:05d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(save_dir / f"heatmap_{safe_query}_lvl{chosen_lvl}.png"),
                cv2.cvtColor(
                    (heatmap_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                ),
            )

            # Create mask from threshold
            output = relev_map[chosen_lvl, k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * 2 - 1
            output = torch.clip(output, 0, 1)
            mask_pred = (output.cpu().numpy() > mask_thresh).astype(np.uint8)

            # Smooth mask
            scale = 5
            kernel = np.ones((scale, scale), np.uint8)
            mask_pred = cv2.morphologyEx(mask_pred, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(
                str(save_dir / f"mask_{safe_query}_lvl{chosen_lvl}.png"),
                mask_pred * 255,
            )

            # Create composited image if RGB available
            if rgb_img is not None:
                composited = rgb_img.clone()
                mask_3ch = np.repeat(mask_pred[..., None], 3, axis=-1)
                mask_3ch = torch.from_numpy(mask_3ch).float().to(device)
                heatmap_overlay = torch.from_numpy(
                    heatmap_img
                ).float().to(device)
                composited = torch.where(
                    mask_3ch > 0.5,
                    heatmap_overlay * 0.7 + composited * 0.3,
                    composited * 0.3,
                )
                cv2.imwrite(
                    str(save_dir / f"composited_{safe_query}_lvl{chosen_lvl}.png"),
                    cv2.cvtColor(
                        (composited.cpu().numpy() * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR,
                    ),
                )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Query LangSplat with text prompts")
    parser.add_argument("--dataset_name", type=str, default="sofa")
    parser.add_argument("--feat_dir", type=str, default="output")
    parser.add_argument("--ae_ckpt_dir", type=str, default="autoencoder/ckpt")
    parser.add_argument("--output_dir", type=str, default="query_result")
    parser.add_argument(
        "--query", type=str, nargs="+", default=["grey sofa"],
        help="Text query(s) to search for"
    )
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument(
        "--encoder_dims",
        nargs="+",
        type=int,
        default=[256, 128, 64, 32, 3],
    )
    parser.add_argument(
        "--decoder_dims",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 256, 512],
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Optional directory with RGB renders for visualization overlay"
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name

    # Feature directories for 3 levels
    feat_dir = [
        os.path.join(
            args.feat_dir, f"{dataset_name}_{i}", "train/ours_None/renders_npy"
        )
        for i in range(1, 4)
    ]

    # Validate feature directories exist
    for fd in feat_dir:
        if not os.path.isdir(fd):
            print(f"ERROR: Feature directory not found: {fd}")
            print("Make sure you have run 'python render.py -m output/sofa_X --include_feature' for all 3 levels.")
            sys.exit(1)

    output_path = os.path.join(args.output_dir, dataset_name)
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")

    if not os.path.isfile(ae_ckpt_path):
        print(f"ERROR: Autoencoder checkpoint not found: {ae_ckpt_path}")
        sys.exit(1)

    print(f"Dataset: {dataset_name}")
    print(f"Feature dirs: {feat_dir}")
    print(f"AE checkpoint: {ae_ckpt_path}")
    print(f"Query texts: {args.query}")
    print(f"Output: {output_path}")

    query_features(
        feat_dir=feat_dir,
        output_path=output_path,
        ae_ckpt_path=ae_ckpt_path,
        query_texts=args.query,
        image_dir=args.image_dir,
        mask_thresh=args.mask_thresh,
        encoder_dims=args.encoder_dims,
        decoder_dims=args.decoder_dims,
    )

