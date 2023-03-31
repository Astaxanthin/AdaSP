# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY

from fastreid.layers import get_norm

from fastreid.utils.events import get_event_storage

@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            heads_extra_bn,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads
        
        # extra bn for head
        self.heads_extra_bn = heads_extra_bn

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        
        has_extra_bn = cfg.MODEL.BACKBONE.EXTRA_BN
        if has_extra_bn:
            heads_extra_bn = get_norm(cfg.MODEL.BACKBONE.NORM, cfg.MODEL.BACKBONE.FEAT_DIM)
        else:
            heads_extra_bn = None
        
        return {
            'backbone': backbone,
            'heads': heads,
            'heads_extra_bn': heads_extra_bn,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    },
                    'adasp': {
                        'scale': cfg.MODEL.LOSSES.ADASP.SCALE,
                        'temp': cfg.MODEL.LOSSES.ADASP.TEMP,
                        'loss_type': cfg.MODEL.LOSSES.ADASP.TYPE,
                    },
                    'supcon': {
                        'scale': cfg.MODEL.LOSSES.SUPCON.SCALE
                    },
                    'ms': {
                        'scale': cfg.MODEL.LOSSES.MS.SCALE,
                        'scale_pos': cfg.MODEL.LOSSES.MS.SCALE_POS,
                        'scale_neg': cfg.MODEL.LOSSES.MS.SCALE_NEG
                    },
                    'proxyan': {
                        'scale': cfg.MODEL.LOSSES.PROXYAN.SCALE,
                        'num_classes': cfg.MODEL.HEADS.NUM_CLASSES
                    },
                    'ep': {
                        'scale': cfg.MODEL.LOSSES.EP.SCALE
                    },
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        
        if self.heads_extra_bn is not None: 
            features = self.heads_extra_bn(features)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        
        if len(outputs) > 1:
        # fmt: off
            pred_class_logits = outputs['pred_class_logits'].detach()
            cls_outputs       = outputs['cls_outputs']
            
            # Log prediction accuracy
            log_accuracy(pred_class_logits, gt_labels)

        pred_features     = outputs['features']
        #print(pred_features.shape)
        # fmt: on

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']
        #print(loss_names)

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')
        #print(loss_dict)
        if 'AdaSPLoss' in loss_names:
            storage = get_event_storage()
            adasp_kwargs = self.loss_kwargs.get('adasp')
            temp = adasp_kwargs.get('temp')
            loss_func = AdaSPLoss(temp, adasp_kwargs.get('loss_type'))
            loss_dict['loss_adasp'] = loss_func(pred_features, gt_labels)*adasp_kwargs.get('scale')

        else:
            orth_measure = -1
        
        if 'SupConLoss' in loss_names:
            supcon_kwargs = self.loss_kwargs.get('supcon')
            loss_func = SupConLoss()
            loss_dict['loss_supcon'] = loss_func(nn.functional.normalize(pred_features, dim = 1).unsqueeze(1), gt_labels)*supcon_kwargs.get('scale')
       
        if 'MSLoss' in loss_names:
            ms_kwargs = self.loss_kwargs.get('ms')
            loss_func = MultiSimilarityLoss(
                ms_kwargs.get('scale_pos'),
                ms_kwargs.get('scale_neg')
            )
            loss_dict['loss_ms'] = loss_func(nn.functional.normalize(pred_features, dim = 1), gt_labels)*ms_kwargs.get('scale')
        
        if 'ProxyanLoss' in loss_names:
            proxyan_kwargs = self.loss_kwargs.get('proxyan')
            loss_func = ProxyAnchorLoss()
            loss_dict['loss_proxyan'] = loss_func(cls_outputs, gt_labels)*proxyan_kwargs.get('scale')
        
        if 'EPLoss' in loss_names:
            ep_kwargs = self.loss_kwargs.get('ep')
            loss_func = EPHNLoss()
            loss_dict['loss_ep'] = loss_func(pred_features, gt_labels)*ep_kwargs.get('scale')
        
        if 'MPLoss' in loss_names:
            mp_kwargs = self.loss_kwargs.get('mp')
            loss_dict['loss_mp'] = mp_loss(pred_features, gt_labels)*mp_kwargs.get('scale')

       # print(loss_dict)
        return loss_dict
