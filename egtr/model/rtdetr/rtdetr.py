# Modified from https://github.com/naver-ai/egtr/blob/7f87450f32758ed8583948847a8186f2ee8b21e3/model/deformable_detr.py
# and https://github.com/lyuwenyu/RT-DETR/blob/b6bf0200b249a6e35b44e0308b6058f55b99696b/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr.py

from dataclasses import dataclass
from typing import Optional, Tuple
import torch 
import torch.nn as nn
from .backbone import PResNet
from .encoder import HybridEncoder
from .decoder import RTDETRTransformerv2
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel
from transformers.file_utils import ModelOutput


# Copyright 2022 SenseTime and The HuggingFace Inc. team. All rights reserved.
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
class RtDetrConfig(PretrainedConfig):
    model_type = "rt_detr"

    def __init__(
        self,
        # PResNet
        presnet_depth = 50,
        presnet_variant = 'd',
        presnet_freeze_at = 0,
        presnet_return_idx = [1, 2, 3],
        presnet_num_stages = 4,
        presnet_freeze_norm = True,
        presnet_pretrained = False,
        # HybridEncoder
        hybrid_encoder_in_channels = [512, 1024, 2048],
        hybrid_encoder_feat_strides = [8, 16, 32],
        hybrid_encoder_hidden_dim = 256,
        hybrid_encoder_use_encoder_idx = [2],
        hybrid_encoder_num_encoder_layers = 1,
        hybrid_encoder_nhead = 8,
        hybrid_encoder_dim_feedforward = 1024,
        hybrid_encoder_dropout = 0.,
        hybrid_encoder_enc_act = 'gelu',
        hybrid_encoder_expansion = 0.5,
        hybrid_encoder_depth_mult = 1,
        hybrid_encoder_act = 'silu',
        # RTDETRTransformerv2
        rt_detr_transformer_feat_channels = [256, 256, 256],
        rt_detr_transformer_num_levels = 3,
        rt_detr_transformer_num_layers = 6,
        rt_detr_transformer_num_queries = 300,
        rt_detr_transformer_num_denoising = 100,
        rt_detr_transformer_label_noise_ratio = 0.5,
        rt_detr_transformer_box_noise_scale = 1.0,
        rt_detr_transformer_eval_idx = 2,
        rt_detr_transformer_num_points = [4, 4, 4],
        rt_detr_transformer_cross_attn_method = 'default',
        rt_detr_transformer_query_select_method = 'default',
        rt_detr_transformer_eval_spatial_size = [640, 640],
        class_cost=1,
        bbox_cost=5,
        giou_cost=2,
        mask_loss_coefficient=1,
        dice_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        focal_alpha=0.25,
        init_std=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.presnet_depth = presnet_depth
        self.presnet_variant = presnet_variant
        self.presnet_freeze_at = presnet_freeze_at
        self.presnet_return_idx = presnet_return_idx
        self.presnet_num_stages = presnet_num_stages
        self.presnet_freeze_norm = presnet_freeze_norm
        self.presnet_pretrained = presnet_pretrained
        self.hybrid_encoder_in_channels = hybrid_encoder_in_channels
        self.hybrid_encoder_feat_strides = hybrid_encoder_feat_strides
        self.hybrid_encoder_hidden_dim = hybrid_encoder_hidden_dim
        self.hybrid_encoder_use_encoder_idx = hybrid_encoder_use_encoder_idx
        self.hybrid_encoder_num_encoder_layers = hybrid_encoder_num_encoder_layers
        self.hybrid_encoder_nhead = hybrid_encoder_nhead
        self.hybrid_encoder_dim_feedforward = hybrid_encoder_dim_feedforward
        self.hybrid_encoder_dropout = hybrid_encoder_dropout
        self.hybrid_encoder_enc_act = hybrid_encoder_enc_act
        self.hybrid_encoder_expansion = hybrid_encoder_expansion
        self.hybrid_encoder_depth_mult = hybrid_encoder_depth_mult
        self.hybrid_encoder_act = hybrid_encoder_act
        self.rt_detr_transformer_feat_channels = rt_detr_transformer_feat_channels
        self.rt_detr_transformer_feat_strides = hybrid_encoder_feat_strides
        self.rt_detr_transformer_hidden_dim = hybrid_encoder_hidden_dim
        self.rt_detr_transformer_num_levels = rt_detr_transformer_num_levels
        self.rt_detr_transformer_num_layers = rt_detr_transformer_num_layers
        self.num_queries = rt_detr_transformer_num_queries
        self.rt_detr_transformer_num_denoising = rt_detr_transformer_num_denoising
        self.rt_detr_transformer_label_noise_ratio = rt_detr_transformer_label_noise_ratio
        self.rt_detr_transformer_box_noise_scale = rt_detr_transformer_box_noise_scale
        self.rt_detr_transformer_eval_idx = rt_detr_transformer_eval_idx
        self.rt_detr_transformer_num_points = rt_detr_transformer_num_points
        self.rt_detr_transformer_cross_attn_method = rt_detr_transformer_cross_attn_method
        self.rt_detr_transformer_query_select_method = rt_detr_transformer_query_select_method
        self.rt_detr_transformer_eval_spatial_size = rt_detr_transformer_eval_spatial_size

        # duplicated for compatibility with DeformableDetrConfig
        self.d_model = hybrid_encoder_hidden_dim
        self.decoder_layers = rt_detr_transformer_num_layers
        # Hungarian matcher
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Loss coefficients
        self.mask_loss_coefficient = mask_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.focal_alpha = focal_alpha
        self.init_std = init_std

        # ONNX
        self.deploy = False

@dataclass
class RtDetrModelOutput(ModelOutput):
    """
    Base class for outputs of the RT-DETR model.
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Predicted normalized bounding boxes (x_center, y_center, width, height) for all queries.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_attention_queries (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of queries used for decoder self-attention.
        decoder_attention_keys (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of keys used for decoder self-attention.
    """

    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attention_queries: Optional[torch.FloatTensor] = None
    decoder_attention_keys: Optional[torch.FloatTensor] = None


"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
class RtDetr(PreTrainedModel):

    def __init__(self, config: RtDetrConfig,
        backbone: nn.Module = PResNet, 
        encoder: nn.Module = HybridEncoder, 
        decoder: nn.Module = RTDETRTransformerv2, 
    ):
        super().__init__(config)
        self.backbone = backbone(
            config.presnet_depth,
            variant=config.presnet_variant,
            num_stages=config.presnet_num_stages,
            return_idx=config.presnet_return_idx,
            freeze_at=config.presnet_freeze_at,
            freeze_norm=config.presnet_freeze_norm,
            pretrained=config.presnet_pretrained,
        )
        self.encoder = encoder(
            in_channels=config.hybrid_encoder_in_channels,
            feat_strides=config.hybrid_encoder_feat_strides,
            hidden_dim=config.hybrid_encoder_hidden_dim,
            nhead=config.hybrid_encoder_nhead,
            dim_feedforward=config.hybrid_encoder_dim_feedforward,
            dropout=config.hybrid_encoder_dropout,
            enc_act=config.hybrid_encoder_enc_act,
            use_encoder_idx=config.hybrid_encoder_use_encoder_idx,
            num_encoder_layers=config.hybrid_encoder_num_encoder_layers,
            expansion=config.hybrid_encoder_expansion,
            depth_mult=config.hybrid_encoder_depth_mult,
            act=config.hybrid_encoder_act,
        )
        self.decoder = decoder(
            num_classes=config.num_labels,
            hidden_dim=config.rt_detr_transformer_hidden_dim,
            num_queries=config.num_queries,
            feat_channels=config.rt_detr_transformer_feat_channels,
            feat_strides=config.rt_detr_transformer_feat_strides,
            num_levels=config.rt_detr_transformer_num_levels,
            num_points=config.rt_detr_transformer_num_points,
            num_layers=config.rt_detr_transformer_num_layers,
            num_denoising=config.rt_detr_transformer_num_denoising,
            label_noise_ratio=config.rt_detr_transformer_label_noise_ratio,
            box_noise_scale=config.rt_detr_transformer_box_noise_scale,
            eval_spatial_size=config.rt_detr_transformer_eval_spatial_size,
            eval_idx=config.rt_detr_transformer_eval_idx,
            cross_attn_method=config.rt_detr_transformer_cross_attn_method,
            query_select_method=config.rt_detr_transformer_query_select_method,
        )
        
    def forward(self, x, targets=None) -> RtDetrModelOutput:
        backbone_output = self.backbone(x)
        encoder_output = self.encoder(backbone_output)        
        decoder_output = self.decoder(encoder_output, targets)

        if self.config.deploy:
            return (decoder_output["pred_logits"],
                    decoder_output["pred_boxes"],
                    decoder_output["last_hidden_state"],
                    decoder_output["decoder_attention_queries"],
                    decoder_output["decoder_attention_keys"])
        
        return RtDetrModelOutput(
            logits=decoder_output["pred_logits"],
            pred_boxes=decoder_output["pred_boxes"],
            last_hidden_state=decoder_output["last_hidden_state"],
            decoder_attention_queries=decoder_output["decoder_attention_queries"],
            decoder_attention_keys=decoder_output["decoder_attention_keys"],
        )
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 