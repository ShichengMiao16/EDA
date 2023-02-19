# EDA: Recalibrating Features and Regression for Oriented Object Detection

## Introduction

![figure](demo/EDA.jpg)

The objects in remote sensing images are normally densely packed, exhibited in arbitrary 
orientations, and surrounded by complex backgrounds. Over the past few years, great efforts have 
been devoted to developing oriented object detection models to accommodate such data characteristics. 
We argue that an effective detection model hinges on three aspects: feature enhancement, 
feature decoupling for classification and localization, and an appropriate bounding box regression 
scheme. In this article, we instantiate the three aspects on top of the classical Faster R-CNN, with three 
novel components proposed. First, we propose a weighted fusion and refinement (WFR) module, 
which adaptively weighs multi-level features and leverages the attention mechanism to refine the 
fused features. Second, we decouple the RoI (region of interest) features for the subsequent classification 
and localization via a lightweight affine transformation based feature decoupling (ATFD) 
module. Third, we propose a post-classification regression (PCR) module for generating the desired 
quadrilateral bounding boxes. Specifically, PCR predicts the precise vertex location on each side 
of a predicted horizontal box, by simply learning to (i) classify the discretized regression range of 
the vertex, and (ii) revise the vertex location with an offset. We conduct extensive experiments on 
the DOTA, DIOR-R, and HRSC2016 datasets to validate the effectiveness of our method.

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Getting Started

Please refer to [getting_started.md](docs/getting_started.md) for the basic usage.

## Acknowledgement

The implementation of VRDet is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [BboxToolkit](https://github.com/jbwang1997/BboxToolkit).