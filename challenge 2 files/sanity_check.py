"""
Sanity check: load a submission's trained weights via predict.py/model.py and
run predictions on synthetic random-noise images at several different sizes
(including non-square ones) to confirm the model handles arbitrary input
shapes correctly, without needing any real dataset on disk.

Run:
  python sanity_check.py <team_name>
  python sanity_check.py <team_name> --images-per-size 5 --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from labels import IDX_TO_NAME
from evaluate import load_submission, IMAGENET_MEAN, IMAGENET_STD

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Deliberately includes non-square shapes to exercise the model's
# pad-to-square + resize-to-224 logic, not just plain resizing.
FIXED_TEST_SIZES = [
    (224, 224),
    (128, 128),
    (96, 160),
    (300, 200),
    (500, 500),
]

# Randomly varied (height, width) pairs, generated fresh per run/seed, to
# additionally stress arbitrary/native-looking shapes including small ones.
NUM_RANDOM_SIZES = 5
RANDOM_SIZE_RANGE = (40, 500)


def generate_random_image(rng: np.random.Generator, height: int, width: int) -> Image.Image:
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def build_batch(images, size=None):
    """Builds a normalized batch tensor. If size is None, images are assumed
    to already share the same shape (no resize applied)."""
    steps = []
    if size is not None:
        steps.append(transforms.Resize(size))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    transform = transforms.Compose(steps)
    return torch.stack([transform(img) for img in images])


def run_sanity_check(team_dir: Path, images_per_size: int, seed: int):
    print(f"Loading submission: {team_dir.name}")
    model = load_submission(team_dir)
    print("Model loaded.\n")

    rng = np.random.default_rng(seed)

    print("Synthetic random-noise images at fixed sizes (no ground truth, just checking it runs):")
    print(f"{'Size (HxW)':<12} {'Output shape':<16} {'Predicted classes'}")
    print("-" * 70)
    for height, width in FIXED_TEST_SIZES:
        images = [generate_random_image(rng, height, width) for _ in range(images_per_size)]
        x = build_batch(images, size=None)
        with torch.no_grad():
            preds = model.predict(x)
        pred_names = [IDX_TO_NAME[p] for p in preds.tolist()]
        print(f"{f'{height}x{width}':<12} {str(list(x.shape)):<16} {pred_names}")

    print(f"\nSynthetic random-noise images at random sizes in {RANDOM_SIZE_RANGE} (one at a time):")
    print(f"{'Size (HxW)':<12} {'Output shape':<16} {'Predicted class'}")
    print("-" * 70)
    for _ in range(NUM_RANDOM_SIZES):
        height = int(rng.integers(*RANDOM_SIZE_RANGE))
        width = int(rng.integers(*RANDOM_SIZE_RANGE))
        image = generate_random_image(rng, height, width)
        x = build_batch([image], size=None)
        with torch.no_grad():
            pred = model.predict(x).item()
        print(f"{f'{height}x{width}':<12} {str(list(x.shape)):<16} {IDX_TO_NAME[pred]}")

    print("\nNo crashes across any tested size -- model accepts arbitrary input shapes.")


def resolve_team_dir(team_name: str | None) -> Path:
    if team_name:
        team_dir = DEFAULT_SUBMISSIONS_DIR / team_name
        if not team_dir.exists():
            print(f"[FAIL] Submission folder not found: {team_dir}")
            sys.exit(1)
        return team_dir

    team_dirs = sorted(d for d in DEFAULT_SUBMISSIONS_DIR.iterdir() if d.is_dir())
    if len(team_dirs) != 1:
        print(
            f"[FAIL] Found {len(team_dirs)} submission folders in {DEFAULT_SUBMISSIONS_DIR}. "
            "Pass a team_name explicitly."
        )
        sys.exit(1)
    return team_dirs[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_name", nargs="?", help="Submission folder name inside submissions/")
    parser.add_argument("--images-per-size", type=int, default=4, help="Number of random images to generate per fixed size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible synthetic images")
    args = parser.parse_args()

    team_dir = resolve_team_dir(args.team_name)

    try:
        run_sanity_check(team_dir, args.images_per_size, args.seed)
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
