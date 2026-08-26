"""
src/evaluation/video_utils.py

Utility to extract frames from a video file into a folder of images,
so evaluate.py (which works on frame folders) can be used on .mp4 outputs.

Usage (run from repo root, like every other script in this repo):
    python -m src.evaluation.video_utils --video path/to/output.mp4 --out_dir path/to/frames
"""
import argparse
import os
import cv2


def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f"{idx:04d}.png")
        cv2.imwrite(out_path, frame)  # frame is already BGR, cv2.imwrite expects BGR
        idx += 1
    cap.release()
    print(f"Extracted {idx} frames to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    extract_frames(args.video, args.out_dir)


if __name__ == "__main__":
    main()