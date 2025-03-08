import os
import argparse
from pathlib import Path

FPS = 30
TIME_STEP = 1 / FPS

args = argparse.ArgumentParser(description='Generate association file for ORBSLAM3')
args.add_argument('--replica_path', type=Path, required=True, help='Replica directory')
args = args.parse_args()


assert args.replica_path.exists(), f"Path {args.replica_path} does not exist"
assert (args.replica_path / "data").exists(), f"Path {args.replica_path / 'data'} does not exist"

scans = sorted([f for f in os.listdir(args.replica_path / "data") if os.path.isdir(args.replica_path / "data" / f)])
assert len(scans) == 18, f"Replica expected to have 18 scans, found {len(scans)}"

for scan in scans:
    image_path = args.replica_path / "data" / scan / "sequence"
    dir_list = os.listdir(image_path)
    rgb_list = sorted([f for f in dir_list if f.endswith("color.jpg")])
    depth_list = sorted([f for f in dir_list if f.endswith("depth.pgm")])

    assert len(rgb_list) == len(depth_list), f"Number of RGB {len(rgb_list)} and depth {len(depth_list)} images do not match for {scan}"

    association = []
    timestamp = 0.0
    for i, (rgb, depth) in enumerate(zip(rgb_list, depth_list)):
        association.append(f"{timestamp:.6f} sequence/{rgb} {timestamp:.6f} sequence/{depth}")
        timestamp += TIME_STEP

    with open(args.replica_path / "data" / scan / "association.txt", "w") as f:
        f.write("\n".join(association))