# Modified from https://github.com/naver-ai/egtr/blob/7f87450f32758ed8583948847a8186f2ee8b21e3/train_egtr.py
# EGTR
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader

from data.open_image import OIDataset, oi_get_statistics
from data.visual_genome import VGDataset, vg_get_statistics
from data.threerscan import ThreeRScanDataset, threerscan_get_statistics
from lib.evaluation.coco_eval import CocoEvaluator
from lib.evaluation.oi_eval import OIEvaluator
from lib.evaluation.sg_eval import (
    BasicSceneGraphEvaluator,
    calculate_mR_from_evaluator_list,
)
from model.rtdetr.feature_extractor import RtDetrFeatureExtractor, RtDetrFeatureExtractorWithAugmentorNoCrop
from model.rtdetr.rtdetr import RtDetrConfig
from model.rtdetr_egtr import RtDetrForSceneGraphGeneration
from util.misc import use_deterministic_algorithms

from train_egtr import evaluate_batch

seed_everything(42, workers=True)

def collate_fn(batch, feature_extractor):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt"
    )
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


class SGG(pl.LightningModule):
    def __init__(
        self,
        architecture,
        auxiliary_loss,
        lr,
        lr_backbone,
        lr_initialized,
        weight_decay,
        pretrained,
        main_trained,
        from_scratch,
        id2label,
        rel_loss_coefficient,
        smoothing,
        rel_sample_negatives,
        rel_sample_nonmatching,
        rel_categories,
        multiple_sgg_evaluator,
        multiple_sgg_evaluator_list,
        single_sgg_evaluator,
        single_sgg_evaluator_list,
        coco_evaluator,
        oi_evaluator,
        feature_extractor,
        num_queries,
        ce_loss_coefficient,
        rel_sample_negatives_largest,
        rel_sample_nonmatching_largest,
        use_freq_bias,
        fg_matrix,
        use_log_softmax,
        freq_bias_eps,
        connectivity_loss_coefficient,
        logit_adjustment,
        logit_adj_tau,
    ):

        super().__init__()
        # replace COCO classification head with custom head
        config = RtDetrConfig()
        config.architecture = architecture
        config.auxiliary_loss = auxiliary_loss
        config.from_scratch = from_scratch
        config.num_rel_labels = len(rel_categories)
        config.num_labels = max(id2label.keys()) + 1
        config.num_queries = num_queries
        config.rel_loss_coefficient = rel_loss_coefficient
        config.smoothing = smoothing
        config.rel_sample_negatives = rel_sample_negatives
        config.rel_sample_nonmatching = rel_sample_nonmatching
        config.ce_loss_coefficient = ce_loss_coefficient
        config.pretrained = pretrained
        config.rel_sample_negatives_largest = rel_sample_negatives_largest
        config.rel_sample_nonmatching_largest = rel_sample_nonmatching_largest

        config.connectivity_loss_coefficient = connectivity_loss_coefficient
        config.use_freq_bias = use_freq_bias
        config.use_log_softmax = use_log_softmax
        config.freq_bias_eps = freq_bias_eps

        config.logit_adjustment = logit_adjustment
        config.logit_adj_tau = logit_adj_tau
        self.config = config

        if config.from_scratch:
            config.presnet_pretrained = True

        self.model = RtDetrForSceneGraphGeneration(config=config, fg_matrix=fg_matrix)
        self.initialized_keys = []
        if not config.from_scratch:
            if pretrained:
                checkpoint = torch.load(pretrained, map_location="cpu")
                
                if 'ema' in checkpoint:
                    state = checkpoint['ema']['module']
                else:
                    state = checkpoint['model']
                
                # ignore num class mismatch
                for k in self.model.obj_det_model.state_dict().keys():
                    if state[k].size() != self.model.obj_det_model.state_dict()[k].size():
                        self.initialized_keys.append(k)
                        print(f"Size mismatch: {k} expected shape {self.model.obj_det_model.state_dict()[k].size()} but get {state[k].size()}")
                state={k:state[k] if state[k].size()==self.model.obj_det_model.state_dict()[k].size() else self.model.obj_det_model.state_dict()[k] for k in self.model.obj_det_model.state_dict().keys()}
                self.model.obj_det_model.load_state_dict(state)

                # egtr head to initialized keys
                for k, v in self.model.egtr_head.named_parameters():
                    if v.requires_grad:
                        self.initialized_keys.append(k)

        if main_trained:
            state_dict = torch.load(main_trained, map_location="cpu")["state_dict"]
            for k in list(state_dict.keys()):
                state_dict[k[6:]] = state_dict.pop(k)  # "model."
            self.model.load_state_dict(state_dict, strict=False)

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.lr_initialized = lr_initialized
        self.weight_decay = weight_decay
        self.multiple_sgg_evaluator = multiple_sgg_evaluator
        self.multiple_sgg_evaluator_list = multiple_sgg_evaluator_list
        self.single_sgg_evaluator = single_sgg_evaluator
        self.single_sgg_evaluator_list = single_sgg_evaluator_list
        self.coco_evaluator = coco_evaluator
        self.oi_evaluator = oi_evaluator
        self.feature_extractor = feature_extractor

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(
            pixel_values=pixel_values,
        )
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        outputs = self.model(
            pixel_values=pixel_values,
            labels=labels,
        )
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        del outputs
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "training_loss": loss.item(),
        }
        log_dict.update({f"training_{k}": v.item() for k, v in loss_dict.items()})
        self.log_dict(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        loss_dict["loss"] = loss
        del loss
        return loss_dict

    def validation_epoch_end(self, outputs):
        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "epoch": torch.tensor(self.current_epoch, dtype=torch.float32),
        }
        for k in outputs[0].keys():
            log_dict[f"validation_" + k] = (
                torch.stack([x[k] for x in outputs]).mean().item()
            )
        self.log_dict(log_dict, on_epoch=True)

    @rank_zero_only
    def on_train_start(self) -> None:
        self.config.save_pretrained(self.logger.log_dir)
        return super().on_train_start()

    def test_step(self, batch, batch_idx):
        # get the inputs
        self.model.eval()

        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        targets = [{k: v.cpu() for k, v in label.items()} for label in batch["labels"]]

        with torch.no_grad():
            outputs = self.forward(pixel_values, pixel_mask)
            # eval SGG
            evaluate_batch(
                outputs,
                targets,
                self.multiple_sgg_evaluator,
                self.multiple_sgg_evaluator_list,
                self.single_sgg_evaluator,
                self.single_sgg_evaluator_list,
                self.oi_evaluator,
                self.config.num_labels,
            )
            # eval OD
            if self.coco_evaluator is not None:
                orig_target_sizes = torch.stack(
                    [target["orig_size"] for target in targets], dim=0
                )
                results = self.feature_extractor.post_process(
                    outputs, orig_target_sizes.to(self.device)
                )  # convert outputs of model to COCO api
                res = {
                    target["image_id"].item(): output
                    for target, output in zip(targets, results)
                }
                self.coco_evaluator.update(res)

    def test_epoch_end(self, outputs):
        log_dict = {}
        # log OD
        if self.coco_evaluator is not None:
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            log_dict.update({"AP50": self.coco_evaluator.coco_eval["bbox"].stats[1]})

        # log SGG
        if self.multiple_sgg_evaluator is not None:
            recall = self.multiple_sgg_evaluator["sgdet"].print_stats()
            mean_recall = calculate_mR_from_evaluator_list(
                self.multiple_sgg_evaluator_list, "sgdet", multiple_preds=True
            )
            log_dict.update(recall)
            log_dict.update(mean_recall)

        if self.single_sgg_evaluator is not None:
            recall = self.single_sgg_evaluator["sgdet"].print_stats()
            mean_recall = calculate_mR_from_evaluator_list(
                self.single_sgg_evaluator_list, "sgdet", multiple_preds=False
            )
            recall = dict(zip(["(single)" + x for x in recall.keys()], recall.values()))
            mean_recall = dict(
                zip(["(single)" + x for x in mean_recall.keys()], mean_recall.values())
            )
            log_dict.update(recall)
            log_dict.update(mean_recall)

        if self.oi_evaluator is not None:
            metrics = self.oi_evaluator.aggregate_metrics()
            log_dict.update(metrics)
        self.log_dict(log_dict, on_epoch=True)
        return log_dict

    def configure_optimizers(self):
        diff_lr_params = ["backbone", "reference_points", "sampling_offsets"]

        if self.lr_initialized is not None:  # rel_predictor
            initialized_lr_params = self.initialized_keys
        else:
            initialized_lr_params = []
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if (not any(nd in n for nd in diff_lr_params))
                    and (not any(nd in n for nd in initialized_lr_params))
                    and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in diff_lr_params) and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        if initialized_lr_params:
            param_dicts.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if any(nd in n for nd in initialized_lr_params)
                        and p.requires_grad
                    ],
                    "lr": self.lr_initialized,
                }
            )
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


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
        "--output_path",
        type=str,
        required=True,
    )

    # Architecture
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--auxiliary_loss", type=str2bool, default=False)
    parser.add_argument(
        "--from_scratch", type=str2bool, default=False
    )  # whether to train without pretrained detr
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
    )  # set to "architecture" when from_scratch is True

    # Hyperparameters
    parser.add_argument("--num_queries", type=int, default=300)
    parser.add_argument("--ce_loss_coefficient", type=float, default=2.0)
    parser.add_argument("--rel_loss_coefficient", type=float, default=15.0)
    parser.add_argument(
        "--connectivity_loss_coefficient", type=float, default=30.0
    )  # OI: 90
    parser.add_argument("--smoothing", type=float, default=1e-14)
    parser.add_argument("--rel_sample_negatives", type=int, default=80)
    parser.add_argument("--rel_sample_nonmatching", type=int, default=80)
    parser.add_argument(
        "--rel_sample_negatives_largest", type=str2bool, default=True
    )  # OI: True
    parser.add_argument(
        "--rel_sample_nonmatching_largest", type=str2bool, default=True
    )  # OI: False

    # Training
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--max_epochs_finetune", type=int, default=25)
    parser.add_argument("--lr_backbone", type=float, default=2e-7)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--lr_initialized", type=float, default=2e-4)  # for pretrained
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)

    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--memo", type=str, default="")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--finetune", type=str2bool, default=True)

    parser.add_argument(
        "--filter_duplicate_rels", type=str2bool, default=True
    )  # for OI
    parser.add_argument("--filter_multiple_rels", type=str2bool, default=True)  # for OI
    parser.add_argument("--use_freq_bias", type=str2bool, default=True)
    parser.add_argument("--use_log_softmax", type=str2bool, default=False)

    # Evaluation
    parser.add_argument("--skip_train", type=str2bool, default=False)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_when_train_end", type=str2bool, default=True)
    parser.add_argument("--eval_single_preds", type=str2bool, default=True)
    parser.add_argument("--eval_multiple_preds", type=str2bool, default=False)

    parser.add_argument("--logit_adjustment", type=str2bool, default=False)
    parser.add_argument("--logit_adj_tau", type=float, default=0.3)

    # Speed up
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])

    args = parser.parse_args()
    if args.from_scratch:
        args.pretrained = args.architecture

    # Feature extractor
    feature_extractor = RtDetrFeatureExtractor.from_pretrained(
        args.architecture, size=800, max_size=1333
    )
    feature_extractor_train = (
        RtDetrFeatureExtractorWithAugmentorNoCrop.from_pretrained(
            args.architecture, size=800, max_size=1333
        )
    )

    # Dataset
    if "3RScan" in args.data_path:
        train_dataset = ThreeRScanDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor_train,
            split="train",
            num_object_queries=args.num_queries,
            debug=args.debug,
        )
        val_dataset = ThreeRScanDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split="val",
            num_object_queries=args.num_queries,
        )
        cats = train_dataset.coco.cats
        id2label = {k - 1: v["name"] for k, v in cats.items()}  # 0 ~ 159 or 19
        fg_matrix = threerscan_get_statistics(train_dataset, must_overlap=True)

    elif "visual_genome" in args.data_path:
        train_dataset = VGDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor_train,
            split="train",
            num_object_queries=args.num_queries,
            debug=args.debug,
        )
        val_dataset = VGDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor,
            split="val",
            num_object_queries=args.num_queries,
        )
        cats = train_dataset.coco.cats
        id2label = {k - 1: v["name"] for k, v in cats.items()}  # 0 ~ 149
        fg_matrix = vg_get_statistics(train_dataset, must_overlap=True)
    else:
        train_dataset = OIDataset(
            data_folder=args.data_path,
            feature_extractor=feature_extractor_train,
            split="train",
            filter_duplicate_rels=args.filter_duplicate_rels,
            filter_multiple_rels=args.filter_multiple_rels,
            num_object_queries=args.num_queries,
            debug=args.debug,
        )
        val_dataset = OIDataset(
            data_folder=args.data_path,
            split="val",
            num_object_queries=args.num_queries,
            feature_extractor=feature_extractor,
        )
        id2label = train_dataset.classes_to_ind  # 0 ~ 600
        fg_matrix = oi_get_statistics(train_dataset, must_overlap=True)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # Evaluator
    rel_categories = train_dataset.rel_categories
    multiple_sgg_evaluator = None
    single_sgg_evaluator = None
    coco_evaluator = None
    oi_evaluator = None

    multiple_sgg_evaluator_list = []
    single_sgg_evaluator_list = []
    if args.eval_when_train_end:
        if args.eval_multiple_preds:
            multiple_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(
                multiple_preds=True
            )  # R@k
            for index, name in enumerate(rel_categories):
                multiple_sgg_evaluator_list.append(
                    (
                        index,
                        name,
                        BasicSceneGraphEvaluator.all_modes(multiple_preds=True),
                    )
                )
        if args.eval_single_preds:
            single_sgg_evaluator = BasicSceneGraphEvaluator.all_modes(
                multiple_preds=False
            )  # R@k
            for index, name in enumerate(rel_categories):
                single_sgg_evaluator_list.append(
                    (
                        index,
                        name,
                        BasicSceneGraphEvaluator.all_modes(multiple_preds=False),
                    )
                )
        if "visual_genome" in args.data_path:
            coco_evaluator = CocoEvaluator(
                val_dataset.coco, ["bbox"]
            )  # initialize evaluator with ground truths
        elif "open-image" in args.data_path:
            oi_evaluator = OIEvaluator(
                train_dataset.rel_categories, train_dataset.ind_to_classes
            )

    # Logger setting
    save_dir = f"{args.output_path}/egtr__{'/'.join(args.pretrained.split('/')[-3:]).replace('/', '__')}"
    if args.from_scratch:
        save_dir += "__from_scratch"
    name = f"batch__{args.batch_size * args.gpus * args.accumulate}__epochs__{args.max_epochs}_{args.max_epochs_finetune}__lr__{args.lr_backbone}_{args.lr}_{args.lr_initialized}"
    if args.memo:
        name += f"__{args.memo}"
    if args.debug:
        name += "__debug"
    if args.resume:
        version = args.version  # for resuming
    else:
        version = None  #  If version is not specified the logger inspects the save directory for existing versions, then automatically assigns the next available version.

    # Trainer setting
    logger = TensorBoardLogger(save_dir, name=name, version=version)
    if os.path.exists(f"{logger.log_dir}/checkpoints"):
        if os.path.exists(f"{logger.log_dir}/checkpoints/last.ckpt"):
            ckpt_path = f"{logger.log_dir}/checkpoints/last.ckpt"
        else:
            ckpt_path = sorted(
                glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
                key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
            )[-1]
    else:
        ckpt_path = None

    # Module
    module = SGG(
        architecture=args.architecture,
        auxiliary_loss=args.auxiliary_loss,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        lr_initialized=args.lr_initialized,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
        main_trained="",
        from_scratch=args.from_scratch,
        id2label=id2label,
        rel_loss_coefficient=args.rel_loss_coefficient,
        smoothing=args.smoothing,
        rel_sample_negatives=args.rel_sample_negatives,
        rel_sample_nonmatching=args.rel_sample_nonmatching,
        rel_categories=rel_categories,
        multiple_sgg_evaluator=multiple_sgg_evaluator,
        multiple_sgg_evaluator_list=multiple_sgg_evaluator_list,
        single_sgg_evaluator=single_sgg_evaluator,
        single_sgg_evaluator_list=single_sgg_evaluator_list,
        coco_evaluator=coco_evaluator,
        oi_evaluator=oi_evaluator,
        feature_extractor=feature_extractor,
        num_queries=args.num_queries,
        ce_loss_coefficient=args.ce_loss_coefficient,
        rel_sample_negatives_largest=args.rel_sample_negatives_largest,
        rel_sample_nonmatching_largest=args.rel_sample_nonmatching_largest,
        use_freq_bias=args.use_freq_bias,
        fg_matrix=fg_matrix,
        use_log_softmax=args.use_log_softmax,
        freq_bias_eps=1e-12,
        connectivity_loss_coefficient=args.connectivity_loss_coefficient,
        logit_adjustment=args.logit_adjustment,
        logit_adj_tau=args.logit_adj_tau,
    )

    # Callback
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        filename="{epoch:02d}-{validation_loss:.2f}",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss", patience=args.patience, verbose=True, mode="min"
    )

    # Train
    trainer = None
    if not args.skip_train:
        # Main training
        if not Path(
            TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            ).log_dir
        ).exists():
            # Training
            trainer = Trainer(
                precision=args.precision,
                logger=logger,
                gpus=args.gpus,
                max_epochs=args.max_epochs,
                gradient_clip_val=args.gradient_clip_val,
                strategy=DDPStrategy(find_unused_parameters=True),
                callbacks=[checkpoint_callback, early_stop_callback],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Main training")
            trainer.fit(module, ckpt_path=ckpt_path)

            try:
                os.chmod(logger.log_dir, 0o0777)
            except PermissionError as e:
                print(e)

        if args.finetune:
            ckpt_path = sorted(  # load best model
                glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
                key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
            )[-1]

            # Finetune trainer setting
            logger = TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            )
            if os.path.exists(f"{logger.log_dir}/checkpoints"):
                finetune_ckpt_path = f"{logger.log_dir}/checkpoints/last.ckpt"
            else:
                finetune_ckpt_path = None

            # Finetune module
            module = SGG(
                architecture=args.architecture,
                auxiliary_loss=args.auxiliary_loss,
                lr=args.lr * 0.1,
                lr_backbone=args.lr_backbone * 0.1,
                lr_initialized=args.lr_initialized * 0.1,
                weight_decay=args.weight_decay,
                pretrained=args.pretrained,
                main_trained=ckpt_path,
                from_scratch=args.from_scratch,
                id2label=id2label,
                rel_loss_coefficient=args.rel_loss_coefficient,
                smoothing=args.smoothing,
                rel_sample_negatives=args.rel_sample_negatives,
                rel_sample_nonmatching=args.rel_sample_nonmatching,
                rel_categories=rel_categories,
                multiple_sgg_evaluator=multiple_sgg_evaluator,
                multiple_sgg_evaluator_list=multiple_sgg_evaluator_list,
                single_sgg_evaluator=single_sgg_evaluator,
                single_sgg_evaluator_list=single_sgg_evaluator_list,
                coco_evaluator=coco_evaluator,
                oi_evaluator=oi_evaluator,
                feature_extractor=feature_extractor,
                num_queries=args.num_queries,
                ce_loss_coefficient=args.ce_loss_coefficient,
                rel_sample_negatives_largest=args.rel_sample_negatives_largest,
                rel_sample_nonmatching_largest=args.rel_sample_nonmatching_largest,
                use_freq_bias=args.use_freq_bias,
                fg_matrix=fg_matrix,
                use_log_softmax=args.use_log_softmax,
                freq_bias_eps=1e-12,
                connectivity_loss_coefficient=args.connectivity_loss_coefficient,
                logit_adjustment=args.logit_adjustment,
                logit_adj_tau=args.logit_adj_tau,
            )

            # Finetune callback
            checkpoint_callback = ModelCheckpoint(
                monitor="validation_loss",
                filename="{epoch:02d}-{validation_loss:.2f}",
                save_last=True,
            )
            early_stop_callback = EarlyStopping(
                monitor="validation_loss",
                patience=args.patience,
                verbose=True,
                mode="min",
            )

            # Training
            trainer = Trainer(
                precision=args.precision,
                logger=logger,
                max_epochs=args.max_epochs_finetune,
                gpus=args.gpus,
                gradient_clip_val=args.gradient_clip_val,
                strategy=DDPStrategy(find_unused_parameters=True),
                callbacks=[checkpoint_callback, early_stop_callback],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Finetune with smaller lr")
            trainer.fit(module, ckpt_path=finetune_ckpt_path)

        if trainer is not None:
            torch.distributed.destroy_process_group()
            try:
                os.chmod(logger.log_dir, 0o0777)
            except PermissionError as e:
                print(e)

    # Evaluation
    if args.eval_when_train_end and (trainer is None or trainer.is_global_zero):
        if args.skip_train and args.finetune:
            logger = TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            )

        # Load best model
        ckpt_path = sorted(
            glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
            key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
        )[-1]
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        module.model.load_state_dict(state_dict)  # load best model

        # Eval
        trainer = Trainer(
            precision=args.precision, logger=logger, gpus=1, max_epochs=-1
        )
        if "visual_genome" in args.data_path:
            test_dataset = VGDataset(
                data_folder=args.data_path,
                feature_extractor=feature_extractor,
                split=args.split,
                num_object_queries=args.num_queries,
            )
        else:
            test_dataset = OIDataset(
                data_folder=args.data_path,
                split=args.split,
                num_object_queries=args.num_queries,
                feature_extractor=feature_extractor,
            )
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=lambda x: collate_fn(x, feature_extractor),
            batch_size=args.eval_batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
            persistent_workers=True,
        )
        if trainer.is_global_zero:
            print("### Evaluation")
        metric = trainer.test(module, dataloaders=test_dataloader)

        # Save eval metric
        metric = metric[0]
        device = "".join(torch.cuda.get_device_name(0).split()[1:2])
        filename = f'{ckpt_path.replace(".ckpt", "")}__{args.split}__{len(test_dataloader)}__{device}'
        if args.logit_adjustment:
            filename += f"__la_{args.logit_adj_tau}"
        metric["eval_arg"] = args.__dict__
        with open(f"{filename}.json", "w") as f:
            json.dump(metric, f)
        print("metric is saved in", f"{filename}.json")
