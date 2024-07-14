# Bimodal Predicate Refinement with Decoupled Entity-Predicate Representations for Scene Graph Generation

This repository contains the official code implementation for the paper Bimodal Predicate Refinement with Decoupled Entity-Predicate Representations for Scene Graph Generation.

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Train
We provide [scripts](./scripts/train.sh) for training the models

## Device

All our experiments are conducted on one NVIDIA GeForce RTX 3090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Tips

We use the `rel_nms` [operation](./maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) provided by [RU-Net](https://github.com/siml3/RU-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) and [HL-Net](https://github.com/siml3/HL-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) in PredCls and SGCls to filter the predicted relation predicates, which encourages diverse prediction results. 

## Help

Be free to contact me (`guoqing.zhang@bjtu.edu.cn`) if you have any questions!

## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [PE-Net](https://github.com/VL-Group/PENET).
