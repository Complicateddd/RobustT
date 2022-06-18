# -*- coding: utf-8 -*-
# @Author : Shijie Li
# @File : CopyPaste.py
# @Project: RobustT
# @CreateTime : 2022/6/18 17:38:33

import torch
import numpy as np
from detectron2.structures import Boxes

import cv2

CLASS_NAMES = [
"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def visual(data_dict):
	for batch_id in range(len(data_dict)):
		img_gt_info = data_dict[batch_id]
		file_name_id = img_gt_info['image_id']
		img_tensor = img_gt_info['image']
		Boxx = img_gt_info['instances'].gt_boxes
		Score = img_gt_info['instances'].scores.cpu().numpy().copy()
		Class = img_gt_info['instances'].gt_classes.cpu().numpy().copy()

		
		im2show = img_tensor.permute(1,2,0).numpy().copy()

		boxes = Boxx.tensor

		for box_i in range(boxes.shape[0]):
			if Score[box_i]<1:
				color = (0,204,0)
			else:
				color = (0,0,204)
			boxx = tuple(int(np.int(x)) for x in boxes[box_i,:4])
			print(boxx,Class[box_i])
			class_name = CLASS_NAMES[Class[box_i]]
			score = Score[box_i]

			cv2.rectangle(im2show,boxx[:2],boxx[2:4],color,2)
			cv2.putText(im2show, '%s: %.3f' % (class_name, score), (boxx[0], boxx[1] + 15), cv2.FONT_HERSHEY_PLAIN,
				1.0, (0, 0, 255), thickness=1)

		cv2.imwrite('resultee{}.png'.format(file_name_id), im2show)
























def intersect(box_a, box_b):
	""" We resize both tensors to [A,B,2] without new malloc:
	[A,2] -> [A,1,2] -> [A,B,2]
	[B,2] -> [1,B,2] -> [A,B,2]
	Then we compute the area of intersect between box_a and box_b.
	Args:
	  box_a: (tensor) bounding boxes, Shape: [A,4].
	  box_b: (tensor) bounding boxes, Shape: [B,4].
	Return:
	  (tensor) intersection area, Shape: [A,B].
	"""
	A = box_a.size(0)
	B = box_b.size(0)
	max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
		box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
	min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
		box_b[:, :2].unsqueeze(0).expand(A, B, 2))
	inter = torch.clamp((max_xy - min_xy), min=0)
	return inter[:, :, 0] * inter[:, :, 1]

def foreground(box_a, box_b):
	"""Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
	is simply the intersection over union of two boxes.  Here we operate on
	ground truth boxes and default boxes.
	E.g.:
	    A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
	Args:
	    box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
	    box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
	Return:
	    jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
	"""
	# print(box_a.device,box_b.device)
	inter = intersect(box_a, box_b)
	# area_a = ((box_a[:, 2]-box_a[:, 0]) *
	#           (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
	area_b = ((box_b[:, 2]-box_b[:, 0]) *
	(box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
	union = area_b
	return inter / union  # [A,B]


def copy_paste_aug(
	stro_unlabel,
	stro_label,
	threash,
	):
	
	score_rate = []

	assert len(stro_unlabel) == len(stro_label)

	instance_num = len(stro_unlabel)

	for ins_id in range(instance_num):
		
		Un_stro_Instance_info = stro_unlabel[ins_id]['instances']
		# Un_stro_Instance_rpn_info = stro_unlabel[ins_id]['rpn_instances']



		#head
		Un_stro_h,Un_stro_w = Un_stro_Instance_info._image_size
		Un_stro_box_tensor = Un_stro_Instance_info._fields['gt_boxes'].tensor.cpu()
		Un_stro_gt = Un_stro_Instance_info._fields['gt_classes']
		Un_stro_score = Un_stro_Instance_info._fields['scores'].clone()


		# print(Un_stro_score.shape)
		if int(Un_stro_score.shape[0])>0:
			score_rate.append(float(torch.mean(Un_stro_score)))

		# rpn
		# Un_stro_box_rpn_tensor = Un_stro_Instance_rpn_info._fields['gt_boxes'].tensor.cpu()
		# SP_stro_rpn_gt = Un_stro_Instance_rpn_info._fields['gt_classes'].clone()



		SP_stro_Instance_info = stro_label[ins_id]['instances']
		SP_stro_box = SP_stro_Instance_info._fields['gt_boxes'].clone()
		SP_stro_gt = SP_stro_Instance_info._fields['gt_classes'].clone()


		SP_stro_box.clip((Un_stro_h,Un_stro_w))
		SP_stro_box_tensor = SP_stro_box.tensor


		area = (SP_stro_box_tensor[:,2]-SP_stro_box_tensor[:,0])*(SP_stro_box_tensor[:,3]-SP_stro_box_tensor[:,1])
		area_mask = area >=5

		ori_box_tensor = SP_stro_Instance_info._fields['gt_boxes'].tensor
		area_ori = (ori_box_tensor[:,2]-ori_box_tensor[:,0])*(ori_box_tensor[:,3]-ori_box_tensor[:,1])
		rest_a = area/area_ori
		part_mask = rest_a >= 0.7

		mask = area_mask&part_mask




		SP_stro_box_tensor = SP_stro_box_tensor[mask]
		SP_stro_gt = SP_stro_gt[mask]

		
		# stro_unlabel[ins_id]['rpn_instances'].remove('scores')



		if SP_stro_box_tensor.shape[0]>0 and Un_stro_box_tensor.shape[0]>0:
			iof = foreground(SP_stro_box_tensor, Un_stro_box_tensor)
			max_iof = torch.max(iof,1)[0]
			mask_iof = max_iof<=threash

			copy_box_tensor = SP_stro_box_tensor[mask_iof]
			SP_stro_gt = SP_stro_gt[mask_iof]
			
			if copy_box_tensor.shape[0]>0:
				for copy_id in range(copy_box_tensor.shape[0]):
					xmin, ymin,xmax,ymax = copy_box_tensor[copy_id]
					xmin = int(xmin)
					ymin = int(ymin)
					xmax = int(xmax)
					ymax = int(ymax)
					stro_unlabel[ins_id]['image'][:,ymin:ymax,xmin:xmax] = stro_label[ins_id]['image'][:,ymin:ymax,xmin:xmax].clone()

			if copy_box_tensor.shape[0]>0:
				# print('a',stro_unlabel[ins_id]['instances'])

				#head
				copy_eff_box = Boxes(torch.cat((Un_stro_box_tensor,copy_box_tensor)))
				copy_eff_gt = torch.cat((Un_stro_gt.cpu(),SP_stro_gt))
				
				# rpn
				# copy_eff_rpn_box = Boxes(torch.cat((Un_stro_box_rpn_tensor,copy_box_tensor)))
				# copy_eff_rpn_gt = torch.cat((SP_stro_rpn_gt.cpu(),SP_stro_gt))

				copy_eff_score = torch.cat((Un_stro_score,torch.ones(copy_box_tensor.shape[0],device =Un_stro_score.device )))



				stro_unlabel[ins_id]['instances'].remove('gt_classes')
				stro_unlabel[ins_id]['instances'].remove('gt_boxes')
				stro_unlabel[ins_id]['instances'].remove('scores')

				# stro_unlabel[ins_id]['rpn_instances'].remove('gt_classes')
				# stro_unlabel[ins_id]['rpn_instances'].remove('gt_boxes')

				stro_unlabel[ins_id]['instances'].set('gt_boxes',copy_eff_box)
				stro_unlabel[ins_id]['instances'].set('gt_classes',copy_eff_gt)
				stro_unlabel[ins_id]['instances'].set('scores',copy_eff_score)


				# stro_unlabel[ins_id]['rpn_instances'].set('gt_boxes',copy_eff_rpn_box)
				# stro_unlabel[ins_id]['rpn_instances'].set('gt_classes',copy_eff_rpn_gt)

				# tmp  = stro_unlabel[ins_id]['instances']._fields
				# tmp['gt_boxes']=copy_eff_box
				# tmp['gt_classes']=copy_eff_gt

				# print('b',stro_unlabel[ins_id]['instances'])
	# print(score_rate)

	if len(score_rate)>0:
		# print('score_rate')
		score_rate = sum(score_rate)/len(score_rate)
	else:
		score_rate = 0


	# visual(stro_unlabel)


	return stro_unlabel,stro_label,score_rate



import random
def random_missing_label(
	full_label,
	missing_rate
	):
	
	for ins_id in range(len(full_label)):

		if random.randint(0,1):

			full_label_Instance_info = full_label[ins_id]['instances']

			anno_len = len(full_label_Instance_info)

			anno_gt_class = full_label_Instance_info._fields['gt_classes'].clone()
			anno_boxes = full_label_Instance_info._fields['gt_boxes'].clone()
			
			candi_index = [i for i in range(anno_len)]

			sample_rate = 1-missing_rate

			sample_index = random.sample(candi_index,int(sample_rate*anno_len))

			torch_sample_index = torch.tensor(sample_index,dtype = torch.long, device = anno_boxes.device)

			anno_boxes = anno_boxes[torch_sample_index]
			anno_gt_class = anno_gt_class[torch_sample_index]


			full_label[ins_id]['instances'].remove('gt_classes')
			full_label[ins_id]['instances'].remove('gt_boxes')

			full_label[ins_id]['instances'].set('gt_boxes',anno_boxes)
			full_label[ins_id]['instances'].set('gt_classes',anno_gt_class)

	return full_label



# def Box_jitter(
# 	pred_instance_list,





# 	)