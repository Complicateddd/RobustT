# -*- coding: utf-8 -*-
# @Author : Shijie Li
# @File : rcnn.py
# @Project: RobustT
# @CreateTime : 2022/6/18 17:38:33

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN,WeakSupervisedRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabWeakRCNN(WeakSupervisedRCNN):
    
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)


        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            batch_ = len(gt_instances)
            gt_weak_instances = torch.zeros(batch_,self.weak_num_class,requires_grad=False).to(self.device)
            gt_class_weak = [x.gt_classes for x in gt_instances]

        else:
            gt_instances = None
            # print('------------------------------------None gt_instances')

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch, weak_branch_ouput = None
            )
            weak_loss,_ = self.weak_head(gt_weak_instances,gt_class_weak,features)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(weak_loss)
            
            return losses, [], [], None, []

        elif branch == "sub_supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            # # roi_head lower branch
            weak_loss,weak_supervised_prediction = self.weak_head(gt_weak_instances,gt_class_weak,features,branch)
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch, weak_branch_ouput = weak_supervised_prediction
            )
            detector_losses["loss_cls"] = detector_losses["loss_cls"].sum()
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(weak_loss)
            return losses, [], [], None, []

        elif branch == "unsup_data_weak" or branch == "Jitter_box_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            if branch == 'unsup_data_weak':
                proposals_roih, ROI_predictions = self.roi_heads(
                    images,
                    features,
                    proposals_rpn,
                    targets=None,
                    compute_loss=False,
                    branch=branch,
                )

                return {}, proposals_rpn, proposals_roih, ROI_predictions,[]

            elif branch == 'Jitter_box_data_weak':
                proposals_roih, jitter_pred_instances, ROI_predictions = self.roi_heads(
                    images,
                    features,
                    proposals_rpn,
                    targets=None,
                    compute_loss=False,
                    branch=branch,
                )
                return {}, proposals_rpn, proposals_roih, ROI_predictions,jitter_pred_instances


        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None



@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)


        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None, []

        elif branch == "sub_supervised":
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, None, compute_loss=False
            )
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            return losses, [], [], None

        elif branch == "unsup_data_weak" or branch == "Jitter_box_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            if branch == 'unsup_data_weak':
                proposals_roih, ROI_predictions = self.roi_heads(
                    images,
                    features,
                    proposals_rpn,
                    targets=None,
                    compute_loss=False,
                    branch=branch,
                )

                return {}, proposals_rpn, proposals_roih, ROI_predictions,[]

            elif branch == 'Jitter_box_data_weak':
                proposals_roih, jitter_pred_instances, ROI_predictions = self.roi_heads(
                    images,
                    features,
                    proposals_rpn,
                    targets=None,
                    compute_loss=False,
                    branch=branch,
                )
                return {}, proposals_rpn, proposals_roih, ROI_predictions,jitter_pred_instances


        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results
