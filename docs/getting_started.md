# Getting Started

This page provides basic tutorials about the usage of EDA.
For installation instructions, please refer to [install.md](install.md).

## Prepare datasets

It is recommended to symlink the dataset root to `$EDA/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.
Details of preparing the DOTA, DIOR-R and HRSC2016 datasets are given as follows.

### Prepare DOTA dataset

It is recommended to arrange your initial data in the following structure.

```
EDA
├── data
│   ├── DOTA
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── labelTxt
│   │   ├── val
│   │   │   ├── images
│   │   │   ├── labelTxt
│   │   ├── test
│   │   │   ├── images
```

After preparing the dataset in the above structure, you need to add the dataset paths to the image split config files 
at `$EDA/BboxToolkit/tools/split_configs/dota/`. Then, run `img_split.py` at `$EDA/BboxToolkit/tools/` to split the images.
More details of `img_split.py` can be referred to [USAGE.md](../BboxToolkit/USAGE.md).

```shell
python img_split.py --base_json split_configs/dota/train.json
python img_split.py --base_json split_configs/dota/val.json
python img_split.py --base_json split_configs/dota/test.json
python img_split.py --base_json split_configs/dota/trainval.json
```

After splitting the original DOTA dataset, you will get the following data structure.
```
EDA
├── data
│   ├── split_DOTA
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── val
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── trainval
│   │   │   ├── images
│   │   │   ├── annfiles
```

### Prepare DIOR-R dataset

DIOR-R dataset is an extended version of DIOR annotated with oriented bounding boxes, 
which shares the same images with DIOR. It is recommended to arrange the dataset in the following structure.

```
EDA
├── data
│   ├── DIOR-R
│   │   ├── Annotations
│   │   ├── JPEGImages
│   │   ├── ImageSets
│   │   │   ├── Segmentation
│   │   │   ├── Layout
│   │   │   ├── Main
│   │   │   │   ├── train.txt
│   │   │   │   ├── val.txt
│   │   │   │   ├── test.txt
│   │   │   │   ├── trainval.txt
```

### Prepare HRSC2016 dataset

HRSC2016 dataset is a remote sensing dataset for ship detection, which contains 1061 images with different sizes ranging from 300×300 to 1500×900.
It is recommended to arrange the dataset in the following structure.

```
EDA
├── data
│   ├── HRSC2016
│   │   ├── FullDataSet
│   │   │   ├── AllImages
│   │   │   ├── Annotations
│   │   │   ├── LandMask
│   │   │   ├── Segmentations
│   │   ├── Train
│   │   │   ├── AllImages
│   │   │   ├── Annotations
│   │   │   ├── LandMask
│   │   │   ├── Segmentations
│   │   ├── Test
│   │   │   ├── AllImages
│   │   │   ├── Annotations
│   │   │   ├── LandMask
│   │   │   ├── Segmentations
│   │   ├── ImageSets
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   │   ├── test.txt
│   │   │   ├── trainval.txt
```

## Training and testing of models for oriented object detection

Training and testing of models for oriented object detection are the same as those in mmdetection.
Before running the training and testing commands, please add the dataset path to the config files 
at `$EDA/configs/_base_/dataset`.

### Training

You can use the following commands to train a model.

```shell
# single-gpu training
python tools/train.py ${CONFIG_FILE} [optional arguments]

# multi-gpu training
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

**Note**: The initial learing rate is set to 0.005 with the batch size of 2. 
if your training batch size is different, please remember to change the initial learing rate 
according to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677).

### Testing

You can use the following command to test a model.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```

**Note**: If you want to test a model on DOTA dataset, the following command is able to directly 
generate detection results on the original full images at `${save_dir}`, which can be directly submitted 
to the [DOTA evaluation server](https://captain-whu.github.io/DOTA/evaluation.html).

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --format-only --options save_dir=${SAVE_DIR}
```

## Image demo

We provide demo scripts to test images in DOTA, DIOR-R and HRSC2016 datasets.

### Image demo for DOTA dataset

`$EDA/demo/huge_image_demo.py` is provided to test a single image in DOTA dataset.

```shell
python demo/huge_image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} ${SPLIT_CONFIG_FILE} \
	 [--device ${GPU_ID}] [--score-thr ${SCORE_THR}]
```

### Image demo for DIOR-R and HRSC2016 datasets

`$EDA/demo/image_demo.py` is provided to test a single image in DIOR-R and HRSC2016 datasets.

```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} \
	 [--device ${GPU_ID}] [--score-thr ${SCORE_THR}]
```
