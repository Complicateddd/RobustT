# Robust Teacher : Self-Correcting Pseudo-Labels Guided Robust Semi-Supervised Learning for Object Detection

#### <u>PyTorch Implements Early Release Version</u> (2022.6)


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

## Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

Follow our specific detectron2 components with [README.md](https://github.com/Complicateddd/RobustT/blob/master/detectron2/README.md) to modify base detection framework.

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
2. We provide the whole dataset info at 



## Training

- Train the Robust Teacher under 1% COCO-supervision

```shell
python train_net.py \
      --num-gpus 2 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup1_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 4 SOLVER.IMG_PER_BATCH_UNLABEL 4
```

- Train the Unbiased Teacher under VOC07 (as labeled set) and VOC12 (as unlabeled set)

```shell
python train_net.py \
      --num-gpus 8 \
      --config configs/voc/voc07_voc12.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8
```

- Train the Unbiased Teacher under VOC07 (as labeled set) and VOC12+COCO20cls (as unlabeled set)

```shell
python train_net.py \
      --num-gpus 8 \
      --config configs/voc/voc07_voc12coco20.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 8 SOLVER.IMG_PER_BATCH_UNLABEL 8
```

## Resume the training

```shell
python train_net.py \
      --resume \
      --num-gpus 2 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 MODEL.WEIGHTS <your weight>.pth
```

## Evaluation

```shell
python train_net.py \
      --eval-only \
      --num-gpus 2 \
      --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1.yaml \
       SOLVER.IMG_PER_BATCH_LABEL 4 SOLVER.IMG_PER_BATCH_UNLABEL 4 MODEL.WEIGHTS <your weight>.pth
```

## Model Weights

MS-COCO:

|  Model  | Supervision |        Batch size         |  AP   |                                       Model Weights                                        |
| :-----: | :---------: | :-----------------------: | :---: | :----------------------------------------------------------------------------------------: |
| R50-FPN |     1%      | 16 labeled + 16 unlabeled | 20.16 | [link](https://drive.google.com/file/d/1NQs5SrQ2-ODEVn_ZdPU_2xv9mxdY6MPq/view?usp=sharing) |
| R50-FPN |     2%      | 16 labeled + 16 unlabeled | 24.16 | [link](https://drive.google.com/file/d/12q-LB4iDvgXGW50Q-bYOahpalUvO3SIa/view?usp=sharing) |
| R50-FPN |     5%      | 16 labeled + 16 unlabeled | 27.84 | [link](https://drive.google.com/file/d/1IJQeRP9wHPU0J27YTea-y3lIW96bMAUu/view?usp=sharing) |
| R50-FPN |     10%     | 16 labeled + 16 unlabeled | 31.39 | [link](https://drive.google.com/file/d/1U9tnJGvzRFSOnOfIHOnelFmlvEfyayha/view?usp=sharing) |

VOC:

|  Model  | Labeled set |  Unlabeled set  |       Batch size        | AP50  |  AP   |                                        Model Weights                                         |
| :-----: | :---------: | :-------------: | :---------------------: | :---: | :---: | :------------------------------------------------------------------------------------------: |
| R50-FPN |    VOC07    |      VOC12      | 8 labeled + 8 unlabeled | 80.51 | 54.48 | [link](https://drive.google.com/drive/folders/1Wo7wGZ2t2sLLJ-HmZ46YOeopPwDwHYPL?usp=sharing) |
| R50-FPN |    VOC07    | VOC12+COCO20cls | 8 labeled + 8 unlabeled | 81.71 | 55.79 | [link](https://drive.google.com/drive/folders/1xSY6nTX2n3RzuTw7dOQ_022RRHffJEPP?usp=sharing) |



