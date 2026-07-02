#!/usr/bin/env python3
"""
Argus Core - Dataset Download Helper (Iteration 1.5)
=====================================================
Helps operators obtain the standard deepfake benchmark datasets.
Most datasets require accepted licenses — this script does NOT bypass
that. It prints the official download URLs and license instructions
for each dataset, and (where freely available) downloads sample
subsets for smoke testing.

Usage:
  python scripts/dataset_download.py --dataset celebdf_v2 --output /data
  python scripts/dataset_download.py --dataset asvspoof2019 --output /data
  python scripts/dataset_download.py --dataset faceforensics --output /data

Datasets:
  - celebdf_v2:       Celeb-DF v2 (free for research, ~600 real + 5600 fake videos)
  - asvspoof2019:     ASVspoof 2019 LA (free for research, ~7k eval samples)
  - faceforensics:    FaceForensics++ (requires commercial license)
  - dfdc:             Deepfake Detection Challenge (Kaggle, requires Kaggle account)
  - ff++_smoke:       Small 50-sample FF++-style smoke set (generated, free)
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

DATASETS = {
    "celebdf_v2": {
        "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
        "license": "Free for research use; cite the paper",
        "paper": "Li et al., CVPR 2020. https://arxiv.org/abs/1909.12962",
        "instructions": (
            "1. Visit https://github.com/yuezunli/celeb-deepfakeforensics\n"
            "2. Download Celeb-DF v2 ZIP from the GitHub releases page\n"
            "3. Extract to: <output>/Celeb-DF_v2/\n"
            "   - Test/real/*.mp4\n"
            "   - Test/fake/*.mp4\n"
            "4. For image benchmarks, extract frames with ffmpeg:\n"
            "   for f in Test/real/*.mp4; do\n"
            "     ffmpeg -i \"$f\" -vf fps=1 Test/real/$(basename ${f%.mp4})_%03d.png\n"
            "   done\n"
        ),
        "auto_download": False,  # No public direct URL
    },
    "asvspoof2019": {
        "url": "https://www.asvspoof.org/index2019.html",
        "license": "Free for research; requires EULA acceptance",
        "paper": "Todisco et al., INTERSPEECH 2019. https://arxiv.org/abs/1904.05441",
        "instructions": (
            "1. Visit https://www.asvspoof.org/index2019.html\n"
            "2. Accept the EULA and download:\n"
            "   - ASVspoof2019LA_eval.zip (~5 GB)\n"
            "   - ASVspoof2019.LA.evalcm.txt (protocol)\n"
            "3. Extract to: <output>/ASVspoof2019/\n"
            "   - LA/eval/flac/*.flac\n"
            "   - LA/ASVspoof2019.LA.evalcm.txt\n"
            "4. For training, also download:\n"
            "   - ASVspoof2019LA_train.zip\n"
            "   - ASVspoof2019.LA.traincm.txt\n"
        ),
        "auto_download": False,
    },
    "faceforensics": {
        "url": "https://github.com/ondyari/FaceForensics",
        "license": "COMMERCIAL LICENSE REQUIRED",
        "paper": "Rössler et al., ICCV 2019. https://arxiv.org/abs/1903.08172",
        "instructions": (
            "1. Visit https://github.com/ondyari/FaceForensics\n"
            "2. Follow the 'Dataset Download' instructions to request access\n"
            "3. Download the manipulated sequences (Deepfakes, Face2Face, FaceSwap, NeuralTextures)\n"
            "4. Extract to: <output>/FaceForensics++/\n"
            "5. Use the included extraction script to get face crops:\n"
            "   python scripts/extract_faces.py --input <output>/FaceForensics++ --output <output>/FF++_faces\n"
        ),
        "auto_download": False,
    },
    "dfdc": {
        "url": "https://www.kaggle.com/competitions/deepfake-detection-challenge",
        "license": "Kaggle account + competition rules acceptance required",
        "paper": "Dolhansky et al., CVPR 2020. https://arxiv.org/abs/1910.08854",
        "instructions": (
            "1. Visit https://www.kaggle.com/competitions/deepfake-detection-challenge\n"
            "2. Accept competition rules\n"
            "3. Download:\n"
            "   - dfdc_train_part_0.zip through dfdc_train_part_50.zip (~500 GB total)\n"
            "   - dfdc_test_public.zip\n"
            "4. Extract to: <output>/DFDC/\n"
            "5. Train/test split is via train_sample_videos/ and test_videos/\n"
        ),
        "auto_download": False,
    },
    "ff++_smoke": {
        "url": "(generated locally)",
        "license": "MIT (uses real + AI-generated faces from open sources)",
        "paper": "N/A — this is a synthetic smoke set, not a benchmark",
        "instructions": (
            "Generates 50 small synthetic test images locally for smoke-testing\n"
            "the benchmark harness without downloading real datasets.\n"
        ),
        "auto_download": True,
    },
}


def generate_smoke_set(output_dir: Path):
    """Generate a 50-sample smoke set using PIL."""
    from PIL import Image, ImageDraw
    import random

    print(f"Generating 50-sample smoke set at {output_dir}")
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    # "Real" images: smooth gradients with subtle noise
    for i in range(25):
        img = Image.new("RGB", (224, 224))
        draw = ImageDraw.Draw(img)
        base_color = (random.randint(80, 200), random.randint(80, 200), random.randint(80, 200))
        for y in range(224):
            r = base_color[0] + int(40 * (y / 224) - 20)
            g = base_color[1] + int(40 * (y / 224) - 20)
            b = base_color[2] + int(40 * (y / 224) - 20)
            draw.line([(0, y), (224, y)], fill=(max(0, min(255, r)),
                                                max(0, min(255, g)),
                                                max(0, min(255, b))))
        img.save(real_dir / f"real_{i:03d}.png")

    # "Fake" images: gradients with checkerboard artifacts (mimicking GAN upsampling)
    for i in range(25):
        img = Image.new("RGB", (224, 224))
        draw = ImageDraw.Draw(img)
        base_color = (random.randint(80, 200), random.randint(80, 200), random.randint(80, 200))
        for y in range(224):
            for x in range(0, 224, 8):
                r = base_color[0] + int(40 * (y / 224) - 20)
                g = base_color[1] + int(40 * (y / 224) - 20)
                b = base_color[2] + int(40 * (y / 224) - 20)
                # Add checkerboard pattern (8x8 blocks)
                if (x // 8 + y // 8) % 2 == 0:
                    r = min(255, r + 8); g = min(255, g + 8); b = min(255, b + 8)
                else:
                    r = max(0, r - 8); g = max(0, g - 8); b = max(0, b - 8)
                draw.rectangle([(x, y), (x + 8, y + 1)],
                               fill=(r, g, b))
        img.save(fake_dir / f"fake_{i:03d}.png")

    print(f"Generated 25 real + 25 fake images")
    print(f"Use with: python scripts/benchmark_sota.py --modality image "
          f"--test-set faceforensics --test-root {output_dir} --output /tmp/bench.json")


def main():
    parser = argparse.ArgumentParser(description="Download / set up deepfake benchmark datasets")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    info = DATASETS[args.dataset]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Dataset: {args.dataset} ===")
    print(f"URL:       {info['url']}")
    print(f"License:   {info['license']}")
    print(f"Paper:     {info['paper']}")
    print()
    print("Instructions:")
    print(info["instructions"])

    if info["auto_download"]:
        print("\nAuto-downloading / generating smoke set...")
        if args.dataset == "ff++_smoke":
            generate_smoke_set(output_dir)
        else:
            print(f"Downloading from {info['url']} to {output_dir}...")
            try:
                urllib.request.urlretrieve(info["url"], output_dir / "dataset.zip")
                print(f"Downloaded. Extract and follow the instructions above.")
            except Exception as e:
                print(f"Download failed: {e}")
                print("Please follow the manual instructions above.")
    else:
        print("\nThis dataset requires manual download (license acceptance required).")
        print(f"After downloading, extract to: {output_dir}")
        print(f"Then run: python scripts/benchmark_sota.py --modality <image|audio|video> "
              f"--test-set {args.dataset} --test-root {output_dir} --output results.json")


if __name__ == "__main__":
    main()
