# Modified from https://github.com/naver-ai/egtr/blob/7f87450f32758ed8583948847a8186f2ee8b21e3/evaluate_egtr.py
# EGTR
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
import argparse
import json
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.open_image import OIDataset
from data.visual_genome import VGDataset
from data.threerscan import ThreeRScanDataset
from lib.evaluation.coco_eval import CocoEvaluator
from lib.evaluation.oi_eval import OIEvaluator
from lib.evaluation.sg_eval import (
    BasicSceneGraphEvaluator,
    calculate_mR_from_evaluator_list,
)
from model.rtdetr.feature_extractor import RtDetrFeatureExtractor
from model.rtdetr.rtdetr import RtDetrConfig
from model.rtdetr_egtr import RtDetrForSceneGraphGeneration
from train_egtr import collate_fn, evaluate_batch


@torch.no_grad()
def calculate_fps(model, dataloader):
    import time
    model.eval()
    for _ in range(100):
        dummy = torch.rand(1, 3, 640, 640).cuda()
        _ = model(
            pixel_values=dummy,
        )

    i = 0
    total_time = 0
    for batch in tqdm(dataloader):
        i += 1
        start = time.time()
        outputs = model(
            pixel_values=batch["pixel_values"].cuda(),
        )
        total_time += time.time() - start
    print(f"FPS: {i / total_time}; Latency: {total_time / i}")


# Reference: https://github.com/facebookresearch/detr/blob/main/engine.py
@torch.no_grad()
def evaluate(
    model,
    dataloader,
    num_labels,
    multiple_sgg_evaluator=None,
    single_sgg_evaluator=None,
    oi_evaluator=None,
    coco_evaluator=None,
    feature_extractor=None,
):
    metric_dict = {}
    model.eval()

    multiple_sgg_evaluator_list = []  # mR@k (for each rel category)
    single_sgg_evaluator_list = []
    if multiple_sgg_evaluator is not None:
        for index, name in enumerate(dataloader.dataset.rel_categories):
            multiple_sgg_evaluator_list.append(
                (index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True))
            )
    if single_sgg_evaluator is not None:
        for index, name in enumerate(dataloader.dataset.rel_categories):
            single_sgg_evaluator_list.append(
                (index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=False))
            )

    final_targets, final_outputs = [], []
    for batch in tqdm(dataloader):
        assert len(batch["pixel_values"]) == 1, "Currently only supports batch size 1"
        outputs = model(
            pixel_values=batch["pixel_values"].cuda(),
        )
        targets = batch["labels"]
        if targets[0]['rel'].sum() > 0:
            evaluate_batch(
                outputs,
                targets,
                multiple_sgg_evaluator,
                multiple_sgg_evaluator_list,
                single_sgg_evaluator,
                single_sgg_evaluator_list,
                oi_evaluator,
                num_labels,
            )
        if coco_evaluator is not None:
            orig_target_sizes = torch.stack(
                [target["orig_size"] for target in targets], dim=0
            )
            results = feature_extractor.post_process(
                outputs, orig_target_sizes.cuda()
            )  # convert outputs of model to COCO api
            assert len(targets) == len(results) == 1
            # res = {
            #     target["image_id"].item(): output
            #     for target, output in zip(targets, results)
            # }
            final_targets.append(targets[0]["image_id"].item())
            final_outputs.append(results[0])

    if coco_evaluator is not None:
        # res[targets[0]["image_id"].item()] = results[0]
        res = {id: output for id, output in zip(final_targets, final_outputs)}
        coco_evaluator.update(res)
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        metric_dict.update({"AP50": coco_evaluator.coco_eval["bbox"].stats[1]})

    if multiple_sgg_evaluator is not None:
        recall = multiple_sgg_evaluator["sgdet"].print_stats()
        mean_recall = calculate_mR_from_evaluator_list(
            multiple_sgg_evaluator_list, "sgdet", multiple_preds=True
        )
        metric_dict.update(recall)
        metric_dict.update(mean_recall)

    if single_sgg_evaluator is not None:
        recall = single_sgg_evaluator["sgdet"].print_stats()
        mean_recall = calculate_mR_from_evaluator_list(
            single_sgg_evaluator_list, "sgdet", multiple_preds=False
        )
        recall = {f"(single){key}": value for key, value in recall.items()}
        mean_recall = {f"(single){key}": value for key, value in mean_recall.items()}
        metric_dict.update(recall)
        metric_dict.update(mean_recall)

    if oi_evaluator is not None:
        metrics = oi_evaluator.aggregate_metrics()
        metric_dict.update(metrics)

    return metric_dict


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument("--data_path", type=str, default="dataset/visual_genome")
    parser.add_argument(
        "--artifact_path",
        type=str,
        required=True,
    )

    # Architecture
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--num_queries", type=int, default=200)

    # Evaluation
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_single_preds", type=str2bool, default=True)
    parser.add_argument("--eval_multiple_preds", type=str2bool, default=False)

    parser.add_argument("--logit_adjustment", type=str2bool, default=False)
    parser.add_argument("--logit_adj_tau", type=float, default=0.3)

    # FPS
    parser.add_argument("--min_size", type=int, default=800)
    parser.add_argument("--max_size", type=int, default=1333)
    parser.add_argument("--infer_only", type=str2bool, default=False)

    # Speed up
    parser.add_argument("--num_workers", type=int, default=4)
    args, unknown = parser.parse_known_args()  # to ignore args when training

    # Feature extractor
    feature_extractor = RtDetrFeatureExtractor.from_pretrained(
        args.architecture, size=args.min_size, max_size=args.max_size
    )

    # Dataset
    if "3RScan" in args.data_path or "Replica" in args.data_path:
        test_dataset = ThreeRScanDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
            num_object_queries=args.num_queries,
        )
        id2label = {k: v["name"] for k, v in test_dataset.coco.cats.items()}  # 0 ~ 159 or 19
        coco_evaluator = CocoEvaluator(
            test_dataset.coco, ["bbox"]
        )
        oi_evaluator = None        
    elif "visual_genome" in args.data_path:
        test_dataset = VGDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
            num_object_queries=args.num_queries,
        )
        id2label = {
            k - 1: v["name"] for k, v in test_dataset.coco.cats.items()
        }  # 0 ~ 149
        coco_evaluator = CocoEvaluator(
            test_dataset.coco, ["bbox"]
        )  # initialize evaluator with ground truths
        oi_evaluator = None
    elif "open-image" in args.data_path:
        test_dataset = OIDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split=args.split,
            num_object_queries=args.num_queries,
        )
        id2label = test_dataset.classes_to_ind  # 0 ~ 600
        oi_evaluator = OIEvaluator(
            test_dataset.rel_categories, test_dataset.ind_to_classes
        )
        coco_evaluator = None

    # Dataloader
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.eval_batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # Evaluator
    multiple_sgg_evaluator = None
    single_sgg_evaluator = None
    if args.eval_multiple_preds:
        multiple_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
    if args.eval_single_preds:
        single_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)

    # Model
    config = RtDetrConfig.from_pretrained(args.artifact_path)
    config.logit_adjustment = args.logit_adjustment
    config.logit_adj_tau = args.logit_adj_tau
    artifact_path = Path(args.artifact_path)
    if os.path.isfile(artifact_path / "rt-detr.engine") and os.path.isfile(artifact_path / "egtr-head.engine"):
        config.deploy = True
        config.obj_det_engine_path = artifact_path / "rt-detr.engine"
        config.egtr_head_engine_path = artifact_path / "egtr-head.engine"
        print("Using TensorRT engine")

    model = RtDetrForSceneGraphGeneration(config=config)

    ckpt_path = sorted(
        glob(f"{args.artifact_path}/checkpoints/epoch=*.ckpt"),
        key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
    )[-1]

    if not config.deploy:
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        model.load_state_dict(state_dict)
        model.cuda()
        model.eval()

    # FPS
    if args.infer_only:
        calculate_fps(model, test_dataloader)
    # Eval
    else:
        metric = evaluate(
            model,
            test_dataloader,
            max(id2label.keys()) + 1,
            multiple_sgg_evaluator,
            single_sgg_evaluator,
            oi_evaluator,
            coco_evaluator,
            feature_extractor,
        )

        # Save eval metric
        device = "".join(torch.cuda.get_device_name(0).split()[1:2])
        filename = f'{ckpt_path.replace(".ckpt", "")}__{args.split}__{len(test_dataloader)}__{device}'
        if args.logit_adjustment:
            filename += f"__la_{args.logit_adj_tau}"
        metric["eval_arg"] = args.__dict__
        with open(f"{filename}.json", "w") as f:
            json.dump(metric, f)
        print("metric is saved in", f"{filename}.json")
