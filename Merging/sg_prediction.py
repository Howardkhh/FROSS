import os
import sys

import torch
import torchvision

sys.path.append("../EGTR")
from model.rtdetr.feature_extractor import RtDetrFeatureExtractor
from model.rtdetr.rtdetr import RtDetrConfig
from model.rtdetr_egtr import RtDetrForSceneGraphGeneration
from lib.pytorch_misc import argsort_desc

class SG_Predictor():
    def __init__(self, args) -> None:
        self.feature_extractor = RtDetrFeatureExtractor.from_pretrained(
        "SenseTime/deformable-detr", size=800, max_size=1333
        )
        self.config = RtDetrConfig.from_pretrained(args.artifact_path)
        self.config.logit_adjustment = False
        self.logit_adj_tau = 0.3

        if os.path.isfile(args.artifact_path / "rt-detr.engine") and os.path.isfile(args.artifact_path / "egtr-head.engine"):
            self.config.deploy = True
            self.config.obj_det_engine_path = args.artifact_path / "rt-detr.engine"
            self.config.egtr_head_engine_path = args.artifact_path / "egtr-head.engine"
            print("Using TensorRT engine")
        else:
            raise ValueError("TensorRT engine not found")
        
        self.model = RtDetrForSceneGraphGeneration(self.config)
        self.model.eval()

        self.obj_thresh = args.obj_thresh
        self.rel_topk = args.rel_topk

        # warm up
        for _ in range(10):
            image = torch.rand(3, 640, 640).cuda()
            image = torchvision.transforms.functional.normalize(
                image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            image = image.unsqueeze(0)
            _ = self.model(image)

    def detect_objects(self, image):
        ori_h, ori_w = image.shape[:2]
        image = torch.tensor(image.copy()).permute(2, 0, 1).cuda()
        image = torchvision.transforms.functional.resize(image, (640, 640))
        image = torchvision.transforms.functional.normalize(
            image / 255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = image.unsqueeze(0)
        obj_det_output = self.model.forward_obj_det(image)
        pred_logits = obj_det_output["logits"][0]
        obj_scores, pred_classes = torch.max(
            pred_logits.softmax(-1), -1
        )

        keep = obj_scores > self.obj_thresh
        pred_classes = pred_classes[keep]
        pred_logits = pred_logits[keep]

        pred_boxes = obj_det_output["pred_boxes"][0][keep]
        pred_boxes[:, ::2] = pred_boxes[:, ::2] * ori_w
        pred_boxes[:, 1::2] = pred_boxes[:, 1::2] * ori_h

        return obj_det_output, obj_scores, pred_classes.cpu().numpy(), pred_logits.softmax(-1).cpu().numpy(), pred_boxes.cpu().numpy().astype(int)

    def extract_relations(self, obj_det_output, obj_scores):
        rel_ext_output = self.model(obj_det_output=obj_det_output)

        keep = obj_scores > self.obj_thresh
        obj_scores = obj_scores[keep]

        sub_ob_scores = torch.outer(obj_scores, obj_scores)
        sub_ob_scores[
            torch.arange(obj_scores.size(0)), torch.arange(obj_scores.size(0))
        ] = 0.0  # prevent self-connection


        pred_rel = torch.clamp(rel_ext_output["pred_rel"][0][keep][:, keep], 0.0, 1.0)
        if "pred_connectivity" in rel_ext_output:
            pred_connectivity = torch.clamp(rel_ext_output["pred_connectivity"][0][keep][:, keep], 0.0, 1.0)
            pred_rel = torch.mul(pred_rel, pred_connectivity)
        
        triplet_scores = torch.mul(pred_rel.max(-1)[0], sub_ob_scores)
        pred_rel_inds = argsort_desc(triplet_scores.cpu().detach().clone().numpy())[
            :self.rel_topk, :
        ]  # [pred_rels, 2(s,o)]
        rel_classes = torch.argmax(pred_rel[pred_rel_inds[:, 0], pred_rel_inds[:, 1]], -1)

        return pred_rel_inds, rel_classes.cpu().numpy()