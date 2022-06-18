# Robust Teacher : Self-Correcting Pseudo-Labels Guided Robust Semi-Supervised Learning for Object Detection

#### *(2022.6) PyTorch Implements Early Version Release*


# Installation

## Prerequisites

- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.

## Install PyTorch in Conda env

```shell
# create conda env
conda create -n detectron2 python=3.6
# activate the enviorment
conda activate detectron2
# install PyTorch >=1.5 with GPU
conda install pytorch torchvision -c pytorch
```



## Clone this repo

```shell
git clone https://github.com/Complicateddd/RobustT.git
```



## Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

***Note:*** Follow our specific detectron2 components with [README.md](https://github.com/Complicateddd/RobustT/blob/master/detectron2/README.md) to modify base detection framework.



## Dataset download

1. Download COCO dataset

```shell
# download images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

2. Download Pascal VOC dataset from [VOC challenges website](http://host.robots.ox.ac.uk:8080/pascal/VOC/).
2. **Update** **your project dataset position** in 'detectron2/data/datasets/builtin.py'
2. **We provide the whole dataset necessary info file, you can download them from**  [VOC_COCO_info](https://pan.baidu.com/s/1jyeyErlD2s314NVh7amllw), password: l2h5



## Training

- Train the Robust Teacher under 10% COCO-supervision

```shell
python train_net.py \
      --num-gpus 2 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1_weak.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 4 SOLVER.IMG_PER_BATCH_UNLABEL 4 \
       OUTPUT_DIR ./output10COCO
```

- Train the Robust Teacher under VOC07 (as labeled set) and VOC12 (as unlabeled set)

```shell
python train_net.py \
      --num-gpus 2 \
      --config configs/voc/weak_super_voc07_voc12.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 4 SOLVER.IMG_PER_BATCH_UNLABEL 4
       OUTPUT_DIR ./outputvoc0712
```

- Train the Robust Teacher under VOC07 (as labeled set) and VOC12+COCO20cls (as unlabeled set)

```shell
python train_net.py \
      --num-gpus 2 \
      --config configs/voc/weak_super_voc07_voc12coco20.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 4 SOLVER.IMG_PER_BATCH_UNLABEL 4 \
       OUTPUT_DIR ./outputvoc0712cococls
```

## Resume the training

```shell
python train_net.py \
      --resume \
      --num-gpus 2 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1_weak.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 \
       MODEL.WEIGHTS <your weight>.pth
```

## Evaluation

```shell
python train_net.py \
      --eval-only \
      --num-gpus 2 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1_weak.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 4 SOLVER.IMG_PER_BATCH_UNLABEL 4 \
       MODEL.WEIGHTS <your weight>.pth
```



## Reference

**This repository draws on the following excellent works:**

[Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher)

[Soft-Teacher](https://github.com/microsoft/SoftTeacher)

