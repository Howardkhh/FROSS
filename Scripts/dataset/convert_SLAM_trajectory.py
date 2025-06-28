import os
import argparse
from pathlib import Path

import numpy as np

args = argparse.ArgumentParser(description='Convert SLAM trajectory to 3RScan format')
args.add_argument('--replica_path', type=Path, required=True, help='Replica directory')
args.add_argument('--orbslam_path', type=Path, required=True, help='ORB_SLAM3 directory')
args = args.parse_args()

assert args.replica_path.exists(), f"Path {args.replica_path} does not exist"
assert (args.replica_path / "data").exists(), f"Path {args.replica_path / 'data'} does not exist"

scans = sorted([f for f in os.listdir(args.replica_path / "data") if os.path.isdir(args.replica_path / "data" / f)])
assert len(scans) == 18, f"Replica expected to have 18 scans, found {len(scans)}"

for scan in scans:
    with open(f"{args.orbslam_path}/CameraTrajectory_{scan}.txt", "r") as f:
        trajectory = f.readlines()

    if len(trajectory) != len([f for f in os.listdir(args.replica_path / "data" / scan / "sequence") if f.endswith("pose.txt") and not f.endswith("slam.pose.txt")]):
        print(f"Warning: Number of frames in trajectory {len(trajectory)} does not match number of frames in data {len(os.listdir(args.replica_path / 'data' / scan / 'sequence'))}")
        exit(1)

    init_pose = np.loadtxt(args.replica_path / "data" / scan / "sequence" / "frame-000000.pose.txt")

    print(f"Processing scene {scan}")
    for idx, traj in enumerate(trajectory):
        timestamp, trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, rot_w = map(float, traj.split())
        # convert to transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = np.array([[1 - 2 * rot_y ** 2 - 2 * rot_z ** 2, 2 * rot_x * rot_y - 2 * rot_z * rot_w, 2 * rot_x * rot_z + 2 * rot_y * rot_w],
                                [2 * rot_x * rot_y + 2 * rot_z * rot_w, 1 - 2 * rot_x ** 2 - 2 * rot_z ** 2, 2 * rot_y * rot_z - 2 * rot_x * rot_w],
                                [2 * rot_x * rot_z - 2 * rot_y * rot_w, 2 * rot_y * rot_z + 2 * rot_x * rot_w, 1 - 2 * rot_x ** 2 - 2 * rot_y ** 2]])
        pose[:3, 3] = np.array([trans_x, trans_y, trans_z])
        pose = init_pose @ pose
        np.savetxt(args.replica_path / "data" / scan / "sequence" / f"frame-{idx:06d}.slam.pose.txt", pose, fmt='%.10f')