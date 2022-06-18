# -*- coding: utf-8 -*-
# @Author : Shijie Li
# @File : roi_heads.py
# @Project: RobustT
# @CreateTime : 2022/6/18 17:38:33

import torch
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from rbteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers

import numpy as np
from detectron2.modeling.poolers import ROIPooler


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
        weak_branch_ouput = None
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:


        # print(weak_branch_ouput)

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:

            losses, _ = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch, weak_branch_ouput
            )
            return proposals, losses
        else:
            if branch=='Jitter_box_data_weak':
                pred_instances, jitter_pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
                return pred_instances, jitter_pred_instances, predictions
            
            else:
                pred_instances, predictions = self._forward_box(
                    features, proposals, compute_loss, compute_val_loss, branch
                )
                return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
        weak_branch_ouput_ = None,
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:

        if branch=='Jitter_box_data_weak':
            features = [features[f] for f in self.box_in_features]
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.box_head(box_features)
            predictions = self.box_predictor(box_features)
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)

            jitter_instances = []

            for base_instance in pred_instances:

                base_box = base_instance.pred_boxes.tensor
                image_size = base_instance.image_size

                jitter_box = self._aug_single(base_box,10,0.08)

                jitter_box = Boxes(jitter_box)
                jitter_box.clip(image_size)

                j_instance = Instances(image_size)
                j_instance.proposal_boxes = jitter_box

                jitter_instances.append(j_instance)

            jitter_with_base_instance = []

            for base_instance,j_instance in zip(pred_instances,jitter_instances):

                base_instance_proposal = Instances(base_instance.image_size)
                base_instance_proposal.proposal_boxes = base_instance.pred_boxes
                jitter_with_base_instance.append(Instances.cat([base_instance_proposal,j_instance]))

            jitter_box_features = self.box_pooler(features, [x.proposal_boxes for x in jitter_with_base_instance])
            jitter_box_features = self.box_head(jitter_box_features)
            jitter_predictions = self.box_predictor(jitter_box_features)
            jitter_pred_instances, _ = self.box_predictor.inference(jitter_predictions, jitter_with_base_instance)

            return pred_instances, jitter_pred_instances, predictions

        else:
            features = [features[f] for f in self.box_in_features]
            
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

            box_features = self.box_head(box_features)

            predictions = self.box_predictor(box_features)
            
            del box_features

            if (
                self.training and compute_loss
            ) or compute_val_loss:  # apply if training loss or val loss

                losses = self.box_predictor.losses(predictions, proposals , branch ,weak_branch_ouput_ )

                if self.train_on_pred_boxes:
                    with torch.no_grad():
                        pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                            predictions, proposals
                        )
                        for proposals_per_image, pred_boxes_per_image in zip(
                            proposals, pred_boxes
                        ):
                            proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
                return losses, predictions
            else:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                return pred_instances, predictions



    def _aug_single(self,box,times=4, frac=0.06):
    # random translate
    # TODO: random flip or something
        box_scale = box[:, 2:4] - box[:, :2]
        box_scale = (
            box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
        )
        aug_scale = box_scale * frac  # [n,4]
        # print(aug_scale)
        offset = (
            torch.randn(times, box.shape[0], 4, device=box.device)
            
        )
        # print(offset)
        offset*= aug_scale[None, ...]

        new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
        return torch.cat(
            [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
        ).view(-1,4)

    
    
    def _forward_get_feature(self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        return box_features,proposals

    def _get_prediction(self,
        predictions,
        proposals,
        compute_loss = True,
        compute_val_loss= False,
        branch=""):
        if (
            self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions


    def _forward_features_one(
        self,
        box_features,
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:

        predictions = self.box_predictor(box_features)

        return predictions

    @torch.no_grad()
    def _forward_features_two(
        self,
        box_features,
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:

        predictions = self.box_predictor(box_features)

        return predictions


    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        # try:
        #     print(gt_boxes)
        # except:
        #     print('ttttttt',len(targets))
        #     print(targets[0])
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            try:
                has_gt = len(targets_per_image) > 0
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            except:
                print(match_quality_matrix) 
                print(targets_per_image.gt_boxes)



            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt
