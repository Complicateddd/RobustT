# -*- coding: utf-8 -*-
# @Author : Shijie Li
# @File : fast_rcnn.py
# @Project: RobustT
# @CreateTime : 2022/6/18 17:38:33

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs,
)


# focal loss
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals, branch, weak_branch_ouput_):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        # print(branch,weak_branch_ouput_)
        scores, proposal_deltas = predictions
        # print(scores.shape,proposal_deltas.shape)

        fl = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        )
        losses = fl.losses(branch,weak_branch_ouput_)

        return losses


class FastRCNNFocalLoss(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        num_classes=80,
    ):
        super(FastRCNNFocalLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes

    def losses(self,branch,weak_branch_ouput_):
        return {
            "loss_cls": self.comput_focal_loss(branch,weak_branch_ouput_),
            "loss_box_reg": self.box_reg_loss(),
        }

    def comput_focal_loss(self,branch,weak_branch_ouput_):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            # print('gt_classes',(self.gt_classes==20).sum())
            # print('pred_class_logits',self.pred_class_logits.shape)

            # print('gt',self.gt_classes)
            total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes ,branch=branch,weak_branch_ouput_ = weak_branch_ouput_)
            total_loss = total_loss / self.gt_classes.shape[0]

            return total_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        gamma=1.0,
        num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target ,branch, weak_branch_ouput_):
        # focal loss
        # balance_weight_dict_tensor = torch.tensor(balance_weight_dict,device = input.device)
        # balance_weight = balance_weight_dict_tensor[target]
        
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE

        if branch=="sub_supervised":
            weak_branch_ouput_ = 1 - weak_branch_ouput_
            balance_weight_tensor = torch.ones(self.num_classes+1,device = input.device)
            # print(input.shape,target,weak_branch_ouput_.shape)
            weak_branch_ouput_ = weak_branch_ouput_.mean(0)
            weak_branch_ouput_ = weak_branch_ouput_ - weak_branch_ouput_.mean()
            weak_branch_ouput_ = weak_branch_ouput_ * 0.05
            balance_weight_tensor[:self.num_classes] = balance_weight_tensor[:self.num_classes] + weak_branch_ouput_
            balance_weight = balance_weight_tensor[target]
            loss = loss * balance_weight

        return loss.sum()

