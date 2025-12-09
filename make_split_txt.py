import argparse
import random
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: Path):
    return sorted([p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def main():
    parser = argparse.ArgumentParser(description="Create train/test split txt for rail UVAD dataset.")
    parser.add_argument("--data-root", default="./data/rail/rail_uvad_dataset",
                        help="Root containing normal_frames/anomaly_frames/anomaly_masks")
    parser.add_argument("--train-out", default="train.txt", help="Output path for train list")
    parser.add_argument("--test-out", default="test.txt", help="Output path for test list")
    parser.add_argument("--train-normal", type=int, default=None,
                        help="Number of normal images for train (rest normals go to test). Default: all normals to train.")
    parser.add_argument("--test-anomaly", type=int, default=None,
                        help="Number of anomaly images for test (default: all anomalies).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling before sampling.")
    args = parser.parse_args()

    root = Path(args.data_root)
    normal_dir = root / "normal_frames"
    anomaly_dir = root / "anomaly_frames"
    mask_dir = root / "anomaly_masks"

    rng = random.Random(args.seed)
    normals = list_images(normal_dir)
    anomalies = list_images(anomaly_dir)

    rng.shuffle(normals)
    rng.shuffle(anomalies)

    # split normals
    if args.train_normal is None or args.train_normal >= len(normals):
        train_normals = normals
        test_normals = []
    else:
        train_normals = normals[:args.train_normal]
        test_normals = normals[args.train_normal:]

    # sample anomalies for test
    if args.test_anomaly is None or args.test_anomaly >= len(anomalies):
        test_anomalies = anomalies
    else:
        test_anomalies = anomalies[:args.test_anomaly]

    # build lines
    # use semicolon separator to be robust to spaces in paths
    train_lines = [f"{img.relative_to(root)};0\n" for img in train_normals]

    test_lines = []
    for img in test_normals:
        test_lines.append(f"{img.relative_to(root)};0\n")

    for img in test_anomalies:
        base = img.stem
        mask_path = ""
        for ext in IMAGE_EXTS:
            cand = mask_dir / (base + ext)
            if cand.exists():
                mask_path = str(cand.relative_to(root))
                break
        if mask_path:
            test_lines.append(f"{img.relative_to(root)};1;{mask_path}\n")
        else:
            test_lines.append(f"{img.relative_to(root)};1\n")

    Path(args.train_out).write_text("".join(train_lines))
    Path(args.test_out).write_text("".join(test_lines))
    print(f"wrote {len(train_lines)} train lines to {args.train_out}")
    print(f"wrote {len(test_lines)} test lines to {args.test_out}")
    print(f"split summary -> train normals: {len(train_normals)}, test normals: {len(test_normals)}, test anomalies: {len(test_anomalies)}, seed={args.seed}")


if __name__ == "__main__":
    main()
