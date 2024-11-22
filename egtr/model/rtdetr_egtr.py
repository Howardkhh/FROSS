# coding=utf-8
# Original sources:
#  - https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/detr/modeling_detr.py
#  - https://github.com/huggingface/transformers/blob/01eb34ab45a8895fbd9e335568290e5d0f5f4491/src/transformers/models/deformable_detr/modeling_deformable_detr.py

# Original code copyright
# Copyright 2021 Facebook AI Research The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright
# EGTR
# Copyright (c) 2024-present NAVER Cloud Corp.
# Apache-2.0

# Modified from https://github.com/naver-ai/egtr/blob/7f87450f32758ed8583948847a8186f2ee8b21e3/model/egtr.py
# FROSS
# Copyright 2024-present *****
# Apache-2.0

import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.detr.feature_extraction_detr import center_to_corners_format
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, requires_backends
from scipy.optimize import linear_sum_assignment

from .rtdetr.rtdetr import RtDetr, RtDetrConfig, RtDetrModelOutput
from .util import (
    dice_loss,
    generalized_box_iou,
    nested_tensor_from_tensor_list,
    sigmoid_focal_loss,
)
from .tensorrt_inference import TRTInference

@dataclass
class RtDetrSceneGraphGenerationOutput(ModelOutput):
    """
    Output type of [`RtDetrForSceneGraphGeneration`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~DetrFeatureExtractor.post_process`] to retrieve the unnormalized bounding
            boxes.
        pred_rel (`torch.FloatTensor` of shape `(batch_size, num_queries, num_queries, num_rel_labels)`):
            Predicted relation scores for all pairs of objects.
        pred_connectivity (`torch.FloatTensor` of shape `(batch_size, num_queries, num_queries, 1)`):
            Predicted connectivity scores for all pairs of objects.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: Optional[torch.FloatTensor] = None
    pred_boxes: Optional[torch.FloatTensor] = None
    pred_rel: Optional[torch.FloatTensor] = None
    pred_connectivity: Optional[torch.FloatTensor] = None
    auxiliary_outputs: Optional[List[Dict]] = None
    

class EgtrHead(PreTrainedModel):
    def __init__(self, config: RtDetrConfig, **kwargs):
        super(EgtrHead, self).__init__(config)
        self.config = config
        self.num_queries = self.config.num_queries
        self.head_dim = config.hybrid_encoder_hidden_dim // config.hybrid_encoder_nhead
        self.layer_head = self.config.rt_detr_transformer_num_layers * config.hybrid_encoder_nhead

        if kwargs.get("fg_matrix", None) is not None:  # when training
            eps = config.freq_bias_eps
            fg_matrix = kwargs.get("fg_matrix", None)
            rel_dist = torch.FloatTensor(
                (fg_matrix.sum(axis=(0, 1))) / (fg_matrix.sum() + eps)
            )
            triplet_dist = torch.FloatTensor(
                fg_matrix + eps / (fg_matrix.sum(2, keepdims=True) + eps)
            )
            if config.use_log_softmax:
                triplet_dist = F.log_softmax(triplet_dist, dim=-1)
            else:
                triplet_dist = triplet_dist.log()
            self.rel_dist = nn.Parameter(rel_dist, requires_grad=False)
            self.triplet_dist = nn.Parameter(triplet_dist, requires_grad=False)
            del rel_dist, triplet_dist
        else:  # when infer
            self.triplet_dist = nn.Parameter(
                torch.Tensor(
                    config.num_labels + 1, config.num_labels + 1, config.num_rel_labels
                ),
                requires_grad=False,
            )
            self.rel_dist = nn.Parameter(
                torch.Tensor(config.num_rel_labels), requires_grad=False
            )

        self.proj_q = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.d_model)
                for i in range(self.config.decoder_layers)
            ]
        )
        self.proj_k = nn.ModuleList(
            [
                nn.Linear(config.d_model, config.d_model)
                for i in range(self.config.decoder_layers)
            ]
        )
        self.final_sub_proj = nn.Linear(config.d_model, config.d_model)
        self.final_obj_proj = nn.Linear(config.d_model, config.d_model)

        self.rel_predictor_gate = nn.Linear(2 * config.d_model, 1)
        self.rel_predictor = DeformableDetrMLPPredictionHead(
            input_dim=2 * config.d_model,
            hidden_dim=config.d_model,
            output_dim=config.num_rel_labels,
            num_layers=3,
        )
        self.connectivity_layer = DeformableDetrMLPPredictionHead(
            input_dim=2 * config.d_model,
            hidden_dim=config.d_model,
            output_dim=1,
            num_layers=3,
        )

    def forward(self, last_hidden_state, logits, decoder_attention_queries, decoder_attention_keys):
        bsz = last_hidden_state.size(0)

        _, num_object_queries, _ = logits.shape
        unscaling = self.head_dim ** 0.5

        # Unscaling & stacking attention queries
        projected_q = []
        for q, proj_q in zip(decoder_attention_queries, self.proj_q):
            projected_q.append(
                proj_q(
                    q.transpose(1, 2).reshape(
                        [bsz, num_object_queries, self.config.d_model]
                    )
                    * unscaling
                )
            )
        decoder_attention_queries = torch.stack(
            projected_q, -2
        )  # [bsz, num_object_queries, num_layers, d_model]
        del projected_q

        # Stacking attention keys
        projected_k = []
        for k, proj_k in zip(decoder_attention_keys, self.proj_k):
            projected_k.append(
                proj_k(
                    k.transpose(1, 2).reshape(
                        [bsz, num_object_queries, self.config.d_model]
                    )
                )
            )
        decoder_attention_keys = torch.stack(
            projected_k, -2
        )  # [bsz, num_object_queries, num_layers, d_model]
        del projected_k

        # Pairwise concatenation
        decoder_attention_queries = decoder_attention_queries.unsqueeze(2).repeat(
            1, 1, num_object_queries, 1, 1
        )
        decoder_attention_keys = decoder_attention_keys.unsqueeze(1).repeat(
            1, num_object_queries, 1, 1, 1
        )
        relation_source = torch.cat(
            [decoder_attention_queries, decoder_attention_keys], dim=-1
        )  # [bsz, num_object_queries, num_object_queries, num_layers, 2*d_model]
        del decoder_attention_queries, decoder_attention_keys

        # Use final hidden representations
        subject_output = (
            self.final_sub_proj(last_hidden_state)
            .unsqueeze(2)
            .repeat(1, 1, num_object_queries, 1)
        )
        object_output = (
            self.final_obj_proj(last_hidden_state)
            .unsqueeze(1)
            .repeat(1, num_object_queries, 1, 1)
        )

        relation_source = torch.cat(
            [
                relation_source,
                torch.cat([subject_output, object_output], dim=-1).unsqueeze(-2),
            ],
            dim=-2,
        )
        del subject_output, object_output

        # Gated sum
        rel_gate = torch.sigmoid(self.rel_predictor_gate(relation_source))
        gated_relation_source = torch.mul(rel_gate, relation_source).sum(dim=-2)
        pred_rel = self.rel_predictor(gated_relation_source)

        # from <Neural Motifs>
        if self.config.use_freq_bias:
            predicted_node = torch.argmax(logits, dim=-1)
            pred_rel += torch.stack(
                [
                    self.triplet_dist[predicted_node[i]][:, predicted_node[i]]
                    for i in range(len(predicted_node))
                ],
                dim=0,
            )

        # Connectivity
        pred_connectivity = self.connectivity_layer(gated_relation_source)
        del gated_relation_source
        del relation_source
        return pred_rel, pred_connectivity, rel_gate

class RtDetrForSceneGraphGeneration(PreTrainedModel):
    def __init__(self, config: RtDetrConfig, **kwargs):
        super(RtDetrForSceneGraphGeneration, self).__init__(config)
        if config.deploy:
            self.obj_det_model = TRTInference(config.obj_det_engine_path, max_batch_size=1)
            self.egtr_head = TRTInference(config.egtr_head_engine_path, max_batch_size=1)
        else:
            self.obj_det_model = RtDetr(config)
            self.egtr_head = EgtrHead(config, **kwargs)

        self.config = config

        if self.config.auxiliary_loss:
            raise ValueError("Auxiliary loss is not supported for RT-DETR.")


    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # outputs_class = outputs_class.transpose(1, 0)
        # outputs_coord = outputs_coord.transpose(1, 0)
        return [
            {"logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
    
    def forward_obj_det(self, pixel_values, labels=None):
        if self.config.deploy:
            obj_det_output = self.obj_det_model({"images": pixel_values})
        else:
            obj_det_output: RtDetrModelOutput = self.obj_det_model(pixel_values, labels)
        return obj_det_output

    def forward(
        self,
        pixel_values=None,
        labels=None,
        obj_det_output=None,
    ):
        assert pixel_values is not None or obj_det_output is not None, "pixel_values or obj_det_output must be provided"
        if self.config.deploy:
            if obj_det_output is None:
                obj_det_output = self.obj_det_model({"images": pixel_values})
            egtr_head_output = self.egtr_head(obj_det_output)
            logits, pred_boxes = obj_det_output["logits"], obj_det_output["pred_boxes"]
            pred_rel, pred_connectivity  = egtr_head_output["pred_rel"], egtr_head_output["pred_connectivity"]
        else:
            if obj_det_output is None:
                obj_det_output: RtDetrModelOutput = self.obj_det_model(pixel_values, labels)
            pred_rel, pred_connectivity, rel_gate = self.egtr_head(
                obj_det_output.last_hidden_state,
                obj_det_output.logits,
                obj_det_output.decoder_attention_queries,
                obj_det_output.decoder_attention_keys,
            )
            logits, pred_boxes = obj_det_output.logits, obj_det_output.pred_boxes
        
        bsz, num_object_queries, _ = logits.shape

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = DeformableDetrHungarianMatcher(
                class_cost=self.config.ce_loss_coefficient,
                bbox_cost=self.config.bbox_cost,
                giou_cost=self.config.giou_cost,
                smoothing=self.config.smoothing,
            )  # the same as loss coefficients
            # Second: create the criterion
            losses = ["labels", "boxes", "relations", "cardinality", "uncertainty"]
            criterion = SceneGraphGenerationLoss(
                matcher=matcher,
                num_object_queries=num_object_queries,
                num_classes=self.config.num_labels,
                num_rel_labels=self.config.num_rel_labels,
                eos_coef=self.config.eos_coefficient,
                losses=losses,
                smoothing=self.config.smoothing,
                rel_sample_negatives=self.config.rel_sample_negatives,
                rel_sample_nonmatching=self.config.rel_sample_nonmatching,
                model_training=self.training,
                focal_alpha=self.config.focal_alpha,
                rel_sample_negatives_largest=self.config.rel_sample_negatives_largest,
                rel_sample_nonmatching_largest=self.config.rel_sample_nonmatching_largest,
            )

            criterion.to(self.device)

            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            outputs_loss["pred_rel"] = pred_rel
            outputs_loss["pred_connectivity"] = pred_connectivity

            loss_dict = criterion(outputs_loss, labels)

            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {
                "loss_ce": self.config.ce_loss_coefficient,
                "loss_bbox": self.config.bbox_loss_coefficient,
            }
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            weight_dict["loss_rel"] = self.config.rel_loss_coefficient
            weight_dict["loss_connectivity"] = self.config.connectivity_loss_coefficient

            loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

            # rel_gate: [bsz, num_objects, num_objects, layer, 1]
            rel_gate = rel_gate.reshape(
                bsz * num_object_queries * num_object_queries, -1
            ).mean(0)
            log_layers = list(
                range(self.config.decoder_layers + 1)
            )  # include final layers

            for i, v in zip(log_layers, rel_gate):
                loss_dict[f"rel_gate_{i}"] = v

        # from <structured sparse rcnn>, post-hoc logit adjustment.
        # reference: https://github.com/google-research/google-research/blob/master/logit_adjustment/main.py#L136-L140
        if self.config.logit_adjustment:
            pred_rel = pred_rel - self.config.logit_adj_tau * self.rel_dist.log().to(
                pred_rel.device
            )

        # Apply sigmoid to logits
        pred_rel = pred_rel.sigmoid()
        pred_connectivity = pred_connectivity.sigmoid()

        return RtDetrSceneGraphGenerationOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            pred_rel=pred_rel,
            pred_connectivity=pred_connectivity,
            auxiliary_outputs=auxiliary_outputs,
        )


# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class SceneGraphGenerationLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        matcher,
        num_object_queries,
        num_classes,
        num_rel_labels,
        eos_coef,
        losses,
        smoothing,
        rel_sample_negatives,
        rel_sample_nonmatching,
        model_training,
        focal_alpha,
        rel_sample_negatives_largest,
        rel_sample_nonmatching_largest,
    ):
        """
        Create the criterion.

        A note on the num_classes parameter (copied from original repo in detr.py): "the naming of the `num_classes`
        parameter of the criterion is somewhat misleading. it indeed corresponds to `max_obj_id + 1`, where max_obj_id
        is the maximum id for a class in your dataset. For example, COCO has a max_obj_id of 90, so we pass
        `num_classes` to be 91. As another example, for a dataset that has a single class with id 1, you should pass
        `num_classes` to be 2 (max_obj_id + 1). For more details on this, check the following discussion
        https://github.com/facebookresearch/detr/issues/108#issuecomment-6config.num_rel_labels269223"

        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            num_classes: number of object categories, omitting the special no-object category.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_object_queries = num_object_queries
        self.num_classes = num_classes
        self.num_rel_labels = num_rel_labels
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.rel_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.rel_sample_negatives = rel_sample_negatives
        self.rel_sample_nonmatching = rel_sample_nonmatching
        self.model_training = model_training
        self.focal_alpha = focal_alpha
        self.rel_sample_negatives_largest = rel_sample_negatives_largest
        self.rel_sample_nonmatching_largest = rel_sample_nonmatching_largest
        self.nonmatching_cost = (
            -torch.log(torch.tensor(1e-8)) * matcher.class_cost
            + 4 * matcher.bbox_cost
            + 2 * matcher.giou_cost
            - torch.log(torch.tensor((1.0 / smoothing) - 1.0))
        )  # set minimum bipartite matching costs for nonmatched object queries
        self.connectivity_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def loss_labels(self, outputs, targets, indices, matching_costs, num_boxes):
        return self._loss_labels_focal(
            outputs, targets, indices, matching_costs, num_boxes
        )

    def _loss_labels_focal(
        self, outputs, targets, indices, matching_costs, num_boxes, log=True
    ):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise ValueError("No logits were found in the outputs")

        source_logits = outputs["logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["class_labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            source_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=source_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [
                source_logits.shape[0],
                source_logits.shape[1],
                source_logits.shape[2] + 1,
            ],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                source_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, matching_costs, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["class_labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    @torch.no_grad()
    def loss_uncertainty(self, outputs, targets, indices, matching_costs, num_boxes):
        nonzero_uncertainty_list = []
        for target, index, matching_cost in zip(targets, indices, matching_costs):
            nonzero_index = target["rel"][index[1], :, :][:, index[1], :].nonzero()
            uncertainty = matching_cost.sigmoid()
            nonzero_uncertainty_list.append(
                uncertainty[nonzero_index[:, 0]] * uncertainty[nonzero_index[:, 1]]
            )
        losses = {"uncertainty": torch.cat(nonzero_uncertainty_list).mean()}
        return losses

    def loss_boxes(self, outputs, targets, indices, matching_costs, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs, "No predicted boxes found in outputs"
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                center_to_corners_format(src_boxes),
                center_to_corners_format(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, matching_costs, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        assert "pred_masks" in outputs, "No predicted masks found in outputs"

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = nn.functional.interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_relations(self, outputs, targets, indices, matching_costs, num_boxes):
        losses = []
        connect_losses = []
        for i, ((src_index, target_index), target, matching_cost) in enumerate(
            zip(indices, targets, matching_costs)
        ):
            # Only calculate relation losses for matched objects (num_object_queries * num_object_queries -> num_obj * num_obj)
            full_index = torch.arange(self.num_object_queries)
            uniques, counts = torch.cat([full_index, src_index]).unique(
                return_counts=True
            )
            full_src_index = torch.cat([src_index, uniques[counts == 1]])
            full_target_index = torch.cat(
                [target_index, torch.arange(len(target_index), self.num_object_queries)]
            )
            full_matching_cost = torch.cat(
                [
                    matching_cost,
                    torch.full(
                        (self.num_object_queries - len(matching_cost),),
                        self.nonmatching_cost,
                        device=matching_cost.device,
                    ),
                ]
            )

            pred_rel = outputs["pred_rel"][i, full_src_index][
                :, full_src_index
            ]  # [num_obj_queries, num_obj_queries, config.num_rel_labels]
            target_rel = target["rel"][full_target_index][
                :, full_target_index
            ]  # [num_obj_queries, num_obj_queries, config.num_rel_labels]

            rel_index = torch.nonzero(target_rel)
            target_connect = torch.zeros(
                target_rel.shape[0], target_rel.shape[1], 1, device=target_rel.device
            )
            target_connect[rel_index[:, 0], rel_index[:, 1]] = 1
            pred_connectivity = outputs["pred_connectivity"][i, full_src_index][
                :, full_src_index
            ]
            loss = self.connectivity_loss(pred_connectivity, target_connect)
            connect_losses.append(loss)

            if self.model_training:
                loss = self._loss_relations(
                    pred_rel,
                    target_rel,
                    full_matching_cost,
                    self.rel_sample_negatives,
                    self.rel_sample_nonmatching,
                )
            else:
                loss = self._loss_relations(
                    pred_rel, target_rel, full_matching_cost, None, None
                )
            losses.append(loss)
        losses = {
            "loss_rel": torch.cat(losses).mean(),
            "loss_connectivity": torch.stack(connect_losses).mean(),
        }
        return losses

    def _loss_relations(
        self,
        pred_rel,
        target_rel,
        matching_cost,
        rel_sample_negatives,
        rel_sample_nonmatching,
    ):
        if (rel_sample_negatives is None) and (rel_sample_nonmatching is None):
            weight = 1.0 - matching_cost.sigmoid()
            weight = torch.outer(weight, weight)
            target_rel = target_rel * weight.unsqueeze(-1)
            loss = self.rel_loss(pred_rel, target_rel).mean(-1).reshape(-1)
        else:
            matched = matching_cost != self.nonmatching_cost
            num_target_objects = sum(matched)

            true_indices = target_rel[
                :num_target_objects, :num_target_objects, :
            ].nonzero()
            false_indices = (
                target_rel[:num_target_objects, :num_target_objects, :] != 1.0
            ).nonzero()
            nonmatching_indices = (
                torch.outer(matched, matched)
                .unsqueeze(-1)
                .repeat(1, 1, self.num_rel_labels)
                != True
            ).nonzero()

            # num_target_relations = len(true_indices)
            if rel_sample_negatives is not None:
                # if rel_sample_negatives == 0 or num_target_relations == 0: # prevent nan loss when no relations
                if rel_sample_negatives == 0:
                    sampled_idx = []
                else:
                    if self.rel_sample_negatives_largest:
                        false_sample_scores = pred_rel[
                            false_indices[:, 0],
                            false_indices[:, 1],
                            false_indices[:, 2],
                        ]
                        sampled_idx = torch.topk(
                            false_sample_scores,
                            min(
                                # num_target_relations * rel_sample_negatives,
                                rel_sample_negatives,
                                false_sample_scores.shape[0],
                            ),
                            largest=True,
                        )[1]
                    else:
                        sampled_idx = torch.tensor(
                            random.sample(
                                range(false_indices.size(0)),
                                min(
                                    # num_target_relations * rel_sample_negatives,
                                    rel_sample_negatives,
                                    false_indices.size(0),
                                ),
                            ),
                            device=false_indices.device,
                        )
                false_indices = false_indices[sampled_idx]
            if rel_sample_nonmatching is not None:
                # if rel_sample_nonmatching == 0 or num_target_relations == 0: # prevent nan loss when no relations
                if rel_sample_nonmatching == 0:
                    sampled_idx = []
                else:
                    if self.rel_sample_nonmatching_largest:
                        nonmatching_sample_scores = pred_rel[
                            nonmatching_indices[:, 0],
                            nonmatching_indices[:, 1],
                            nonmatching_indices[:, 2],
                        ]
                        sampled_idx = torch.topk(
                            nonmatching_sample_scores,
                            min(
                                # num_target_relations * rel_sample_nonmatching,
                                rel_sample_nonmatching,
                                nonmatching_indices.size(0),
                            ),
                            largest=True,
                        )[1]
                    else:
                        sampled_idx = torch.tensor(
                            random.sample(
                                range(nonmatching_indices.size(0)),
                                min(
                                    # num_target_relations * rel_sample_nonmatching,
                                    rel_sample_nonmatching,
                                    nonmatching_indices.size(0),
                                ),
                            ),
                            device=nonmatching_indices.device,
                        )
                nonmatching_indices = nonmatching_indices[sampled_idx]

            relation_indices = torch.cat(
                [true_indices, false_indices, nonmatching_indices]
            )
            pred_rel = pred_rel[
                relation_indices[:, 0], relation_indices[:, 1], relation_indices[:, 2]
            ]
            target_rel = target_rel[
                relation_indices[:, 0], relation_indices[:, 1], relation_indices[:, 2]
            ]

            weight = 1.0 - matching_cost.sigmoid()
            weight = weight[relation_indices[:, 0]] * weight[relation_indices[:, 1]]
            target_rel = target_rel * weight
            loss = self.rel_loss(pred_rel, target_rel)
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, matching_costs, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "relations": self.loss_relations,
            "uncertainty": self.loss_uncertainty,
        }
        assert loss in loss_map, f"Loss {loss} not supported"
        return loss_map[loss](outputs, targets, indices, matching_costs, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "auxiliary_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, matching_costs = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss, outputs, targets, indices, matching_costs, num_boxes
                )
            )

        if "pred_rels" in outputs:
            for pred_rel in outputs["pred_rels"]:
                outputs["pred_rel"] = pred_rel
                _loss_dict = self.loss_relations(
                    outputs, targets, indices, matching_costs, num_boxes
                )
                losses["loss_rel"] += _loss_dict["loss_rel"]

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):

                indices, matching_costs = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss in ["masks", "relations", "uncertainty"]:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(
                        loss,
                        auxiliary_outputs,
                        targets,
                        indices,
                        matching_costs,
                        num_boxes,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["class_labels"] = torch.zeros_like(bt["class_labels"])
            indices, matching_costs = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ["masks", "relations", "uncertainty"]:
                    continue
                l_dict = self.get_loss(
                    loss, enc_outputs, bin_targets, indices, matching_costs, num_boxes
                )
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class DeformableDetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.
    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableDetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        class_cost: float = 1,
        bbox_cost: float = 1,
        giou_cost: float = 1,
        smoothing=0.0,
    ):
        """
        Creates the matcher.

        Params:
            class_cost: This is the relative weight of the classification error in the matching cost
            bbox_cost:
                This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            giou_cost: This is the relative weight of the giou loss of the bounding box in the matching cost
            smoothing: non-negative value for adaptive smoothing (do not apply smoothing if smoothing is set to 0.0)
        """
        super().__init__()

        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        assert (
            class_cost != 0 or bbox_cost != 0 or giou_cost != 0
        ), "All costs of the Matcher can't be 0"
        self.smoothing = smoothing
        self.bias_epsilon = torch.log(torch.tensor(1e-8))

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Performs the matching.

        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                 objects in the target) containing the class labels "boxes": Tensor of dim [num_target_boxes, 4]
                 containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:

                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["logits"].flatten(0, 1).sigmoid()
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["class_labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        )
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = (
            pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        )  # min (1-alpha) * log(1e-8) max alpha * -log(1e-8)

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, tgt_bbox, p=1)  # min 0 max 4

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(
            center_to_corners_format(out_bbox), center_to_corners_format(tgt_bbox)
        )  # min -1 max 1

        # Final cost matrix
        cost_matrix = (
            self.bbox_cost * bbox_cost
            + self.class_cost * class_cost
            + self.giou_cost * giou_cost
        )
        cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()

        if self.smoothing:
            # If object queries are perfectly matched to gt objects,
            # class_cost = (1-alpha) * self.bias_epsilon, bbox_cost = 0, and giou_cost = -1.
            cost_min = (
                self.class_cost * (1 - alpha) * self.bias_epsilon - self.giou_cost
            )
            inverse_sigmoid_smoothing = -torch.log(
                torch.tensor((1.0 / self.smoothing) - 1.0)
            )
            cost_matrix = cost_matrix - cost_min + inverse_sigmoid_smoothing

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(cost_matrix.split(sizes, -1))
        ]

        matching_costs = [
            c[i, indices[i][0], indices[i][1]].to(out_prob.device)
            for i, c in enumerate(cost_matrix.split(sizes, -1))
        ]
        indices = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
        return indices, matching_costs