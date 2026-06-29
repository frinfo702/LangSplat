"""言語特徴量マップ (.npy) を PNG 画像として可視化するユーティリティ"""
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="View language feature .npy as PNG")
    parser.add_argument("input", type=str, help="Path to .npy file or directory")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        npy_files = sorted(input_path.glob("*.npy"))
    else:
        npy_files = [input_path]

    output_dir = Path(args.output) if args.output else input_path.parent / "viewed"
    output_dir.mkdir(parents=True, exist_ok=True)

    for npy_file in npy_files:
        data = np.load(npy_file)  # shape: (H, W, 3)

        # 特徴量マップを 0-255 に正規化
        if data.min() < 0 or data.max() > 1:
            data_min = data.min(axis=(0, 1), keepdims=True)
            data_max = data.max(axis=(0, 1), keepdims=True)
            data = (data - data_min) / (data_max - data_min + 1e-8)

        img = (data * 255).astype(np.uint8)
        Image.fromarray(img).save(output_dir / f"{npy_file.stem}.png")
        print(f"Saved: {output_dir / f'{npy_file.stem}.png'}")

    print(f"\nDone! {len(npy_files)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
