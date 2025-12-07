import argparse
import os
import subprocess
from pathlib import Path


def extract_video(video_path: Path, output_dir: Path, fps: str = None, overwrite: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.png"
    cmd = ["ffmpeg", "-i", str(video_path)]
    if fps:
        cmd.extend(["-vf", f"fps={fps}"])
    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")
    cmd.extend(["-vsync", "0", str(pattern)])
    print(f"[ffmpeg] {video_path.name} -> {output_dir}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos into per-video folders.")
    parser.add_argument("--video-root", required=True, help="Root directory containing videos (e.g., training/videos).")
    parser.add_argument("--output-root", required=True, help="Root directory to save frames (e.g., training/frames).")
    parser.add_argument("--fps", default=None, help="Optional target fps (e.g., 10). If omitted, uses source fps.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frames.")
    args = parser.parse_args()

    video_root = Path(args.video_root)
    output_root = Path(args.output_root)
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    video_files = sorted([p for p in video_root.iterdir() if p.suffix.lower() in {".avi", ".mp4", ".mkv"}])
    if not video_files:
        raise RuntimeError(f"No videos found in {video_root}")

    for vid in video_files:
        out_dir = output_root / vid.stem
        extract_video(vid, out_dir, fps=args.fps, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
