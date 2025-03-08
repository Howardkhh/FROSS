import os
import glob
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pyvips

@dataclass
class CameraIntrinsic():
    fx: float
    fy: float
    cx: float
    cy: float


class SG_Loader():
    def __init__(self, scan_id, split, args, gt_sg=False):
        self.scan_id = scan_id
        self.split = split
        self.threerscan_path = Path(args.dataset_path) / "data"

        self.color_frame_names = os.listdir(self.threerscan_path / scan_id / "sequence")
        self.color_frame_names = [f for f in self.color_frame_names if f.endswith(".color.jpg") and not f.endswith("rendered.color.jpg")]
        self.color_frame_names.sort()

        depth_file_ext = ".rendered.depth.png" if args.label_categories == "scannet" else ".depth.pgm"
        self.depth_frame_names = os.listdir(self.threerscan_path / scan_id / "sequence")
        self.depth_frame_names = [f for f in self.depth_frame_names if f.endswith(depth_file_ext)]
        self.depth_frame_names.sort()

        info_file = self.threerscan_path / scan_id / "sequence" / "_info.txt"
        with open(info_file) as f:
            infos = f.readlines()
        for info in infos:
            if info.startswith("m_colorWidth = "):
                color_w = int(info[len("m_colorWidth = "):])
            if info.startswith("m_colorHeight = "):
                color_h = int(info[len("m_colorHeight = "):])
            if info.startswith("m_depthWidth = "):
                depth_w = int(info[len("m_depthWidth = "):])
            if info.startswith("m_depthHeight = "):
                depth_h = int(info[len("m_depthHeight = "):])
            if info.startswith("m_depthShift = "):
                self.depth_shift = int(info[len("m_depthShift = "):])
            if info.startswith("m_calibrationColorIntrinsic = "):
                color_intrinsic = info[len("m_calibrationColorIntrinsic = "):].strip().split()
                self.color_intrinsic = CameraIntrinsic(fx=float(color_intrinsic[0]), fy=float(color_intrinsic[5]), cx=float(color_intrinsic[2]), cy=float(color_intrinsic[6]))
            if info.startswith("m_calibrationDepthIntrinsic = "):
                depth_intrinsic = info[len("m_calibrationDepthIntrinsic = "):].strip().split()
                self.depth_intrinsic = CameraIntrinsic(fx=float(depth_intrinsic[0]), fy=float(depth_intrinsic[5]), cx=float(depth_intrinsic[2]), cy=float(depth_intrinsic[6]))

        self.colorSize = (color_h, color_w)

        if args.label_categories == "scannet": # 90 degrees CW rotation
            self.colorSize = (self.colorSize[1], self.colorSize[0])
            self.color_intrinsic.fx, self.color_intrinsic.fy = self.color_intrinsic.fy, self.color_intrinsic.fx
            self.color_intrinsic.cx, self.color_intrinsic.cy = color_h - self.color_intrinsic.cy, self.color_intrinsic.cx
            self.depth_intrinsic.fx, self.depth_intrinsic.fy = self.depth_intrinsic.fy, self.depth_intrinsic.fx
            self.depth_intrinsic.cx, self.depth_intrinsic.cy = depth_h - self.depth_intrinsic.cy, self.depth_intrinsic.cx

        if args.use_gt_pose:
            pose_files = sorted([f for f in glob.glob(str(self.threerscan_path / scan_id / "sequence" / "*.pose.txt")) if not f.endswith("slam.pose.txt")])
        else:
            pose_files = sorted([f for f in glob.glob(str(self.threerscan_path / scan_id / "sequence" / "*.slam.pose.txt"))])
        assert len(pose_files) == len(self.color_frame_names) == len(self.depth_frame_names), \
            f"Number of pose files ({len(pose_files)}), number of frames ({len(self.color_frame_names)}), and number of depth frames ({len(self.depth_frame_names)}) do not match"

        rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]) # rotation matrix with theta = -90 degrees
        self.camera_rot = np.ndarray((len(pose_files), 3, 3))
        self.camera_trans = np.ndarray((len(pose_files), 3))
        for idx, pose_file in enumerate(pose_files):
            with open(pose_file) as f:
                extrinsic = np.array([list(map(float, line.strip().split())) for line in f])
            self.camera_rot[idx] = extrinsic[:3, :3] 
            if args.label_categories == "scannet": # 90 degrees CW rotation
                self.camera_rot[idx] = self.camera_rot[idx] @ rotation # rotate the camera first, then apply the extrinsic. P_3D = E @ R @ P_2D + T
            self.camera_trans[idx] = extrinsic[:3, 3]

        self.current_idx = 0
        self.sequnce_len = len(self.color_frame_names)

        self.imgs = np.ndarray((self.sequnce_len, self.colorSize[0], self.colorSize[1], 3), dtype=np.uint8)
        self.depths = np.ndarray((self.sequnce_len, self.colorSize[0], self.colorSize[1]), dtype=np.float64)
        for idx in range(self.sequnce_len):
            depth_path = self.threerscan_path / self.scan_id / "sequence" / self.depth_frame_names[idx]
            depth = pyvips.Image.new_from_file(str(depth_path), access="sequential").numpy()
            depth = depth / self.depth_shift # mm to m
            self.depths[idx] = depth

            if gt_sg: continue # Skip loading images if ground truth scene graph is loaded

            img_path = self.threerscan_path / self.scan_id / "sequence" / self.color_frame_names[idx]
            img = pyvips.Image.new_from_file(str(img_path), access="sequential").numpy()
            if args.label_categories == "scannet": # 90 degrees CW rotation
                self.imgs[idx] = np.rot90(img, 3)
            elif args.label_categories == "replica":
                self.imgs[idx] = img


    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= self.sequnce_len:
            raise StopIteration

        img = self.imgs[self.current_idx]
        depth = self.depths[self.current_idx]
        
        assert f"{self.scan_id}-{self.current_idx:06d}.jpg" == f"{self.scan_id}-{self.color_frame_names[self.current_idx][6:12]}.jpg", \
            f"Frame name mismatch: {self.scan_id}-{self.current_idx:06d}.jpg != {self.current_idx}-{self.color_frame_names[self.current_idx][6:12]}.jpg"
        
        camera_rot = self.camera_rot[self.current_idx]
        camera_trans = self.camera_trans[self.current_idx]
        self.current_idx += 1
        return img, depth, camera_rot, camera_trans

class GT_SG_Loader(SG_Loader):
    def __init__(self, scan_id, split, obj_coco, rel_ann, args):
        super().__init__(scan_id, split, args, gt_sg=True)
        
        self.obj_coco = obj_coco
        imgIds = self.obj_coco.getImgIds()
        img_info = self.obj_coco.loadImgs(imgIds)

        self.img_file2id = {img["file_name"]: img["id"] for img in img_info}

        self.rel_ann = rel_ann

    def __next__(self):
        if self.current_idx >= self.sequnce_len:
            raise StopIteration

        # img = self.imgs[self.current_idx]
        depth = self.depths[self.current_idx]
        
        assert f"{self.scan_id}-{self.current_idx:06d}.jpg" == f"{self.scan_id}-{self.color_frame_names[self.current_idx][6:12]}.jpg", \
            f"Frame name mismatch: {self.scan_id}-{self.current_idx:06d}.jpg != {self.current_idx}-{self.color_frame_names[self.current_idx][6:12]}.jpg"
        imgId = self.img_file2id[f"{self.scan_id}-{self.current_idx:06d}.jpg"]
        ann_id = self.obj_coco.getAnnIds(imgId)
        anns = self.obj_coco.loadAnns(ann_id)
        classes = np.array([ann["category_id"]-1 for ann in anns], dtype=np.int64)
        bboxes = np.array([ann["bbox"] for ann in anns], dtype=np.int64)
        if len(bboxes) > 0:
            bboxes[:, :2] += bboxes[:, 2:] // 2

        rel_ann = self.rel_ann[str(imgId)]
        rel_classes = np.array([rel[2] for rel in rel_ann], dtype=np.int64)
        rels = np.array([(rel[0], rel[1]) for rel in rel_ann], dtype=np.int64)
        
        camera_rot = self.camera_rot[self.current_idx]
        camera_trans = self.camera_trans[self.current_idx]
        self.current_idx += 1
        return depth, classes, bboxes, rel_classes, rels, camera_rot, camera_trans
