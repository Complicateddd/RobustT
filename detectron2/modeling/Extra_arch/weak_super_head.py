import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import Extra_ARCH_REGISTRY

from detectron2.config import configurable



@Extra_ARCH_REGISTRY.register()
class ws_head(nn.Module):

    @configurable
    def __init__(self,num_class):
        super().__init__()
        self.num_class = num_class
        self.head_1 = nn.Conv2d(256,256,1,bias = False)
        self.head_2 = nn.Conv2d(256,num_class,1,bias = False)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(num_class)


    @classmethod
    def from_config(cls, cfg):
        return {
        'num_class': cfg.MODEL.ROI_HEADS.NUM_CLASSES
        }

    @torch.no_grad()
    def fetch_image_gt(self,gt_weak_instances,gt_class_weak):
        # print(batched_inputs)
        # GT_class = [x.gt_classes.clone() for x in batched_inputs]
        # batch = len(GT_class)
        # label_tensor = torch.zeros(len_batched_inputs,20,requires_grad=False)
        len_batched_inputs = gt_weak_instances.shape[0]

        for batch_id in range(len_batched_inputs):
            image_label = gt_class_weak[batch_id].tolist()
            for gt_ in image_label:
                gt_weak_instances[batch_id][gt_] = 1.

        return gt_weak_instances

    def forward(self,gt_weak_instances,gt_class_weak,features,branch='supervised'):

        feat_feature = features['p6']

        output = self.head_1(feat_feature)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.head_2(output)
        output = self.bn2(output)

        output = self.gap(F.sigmoid(output))

        output = output.squeeze(-1).squeeze(-1)

        loss={}

        
        multi_label_gt = self.fetch_image_gt(gt_weak_instances,gt_class_weak)
        del gt_weak_instances
        del gt_class_weak
        # multi_label_gt = multi_label_gt.to(output.device)
        loss['loss_weak_supervised'] = F.binary_cross_entropy(output,multi_label_gt,reduction='mean')*0.05
        
        if branch=='supervised':
            return loss,None
        else:
            weak_supervised_prediction = output
            return loss,weak_supervised_prediction


        # if branch == 'supervised':
        #     loss = self.loss(output,)
        #     return None