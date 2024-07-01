# Bimodal Predicate Refinement with Decoupled Entity-Predicate Representations for Scene Graph Generation

This repository contains the official code implementation for the paper Bimodal Predicate Refinement with Decoupled Entity-Predicate Representations for Scene Graph Generation.

## Installation
Check [INSTALL.md](./INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](./DATASET.md) for instructions of dataset preprocessing.

## Train
We provide [scripts](./scripts/train.sh) for training the models
```
export CUDA_VISIBLE_DEVICES=1
export NUM_GUP=1
echo "TRAINING Predcls"

MODEL_NAME='PE-NET_PredCls'
mkdir ./checkpoints/${MODEL_NAME}/
cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py ./checkpoints/${MODEL_NAME}/
cp ./scripts/train.sh ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/relation_head.py ./checkpoints/${MODEL_NAME}

python3 \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 30000 \
  SOLVER.CHECKPOINT_PERIOD 30000 GLOVE_DIR ./datasets/vg/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0;

```

## Device

All our experiments are conducted on one NVIDIA GeForce RTX 3090, if you wanna run it on your own device, make sure to follow distributed training instructions in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Tips

We use the `rel_nms` [operation](./maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) provided by [RU-Net](https://github.com/siml3/RU-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) and [HL-Net](https://github.com/siml3/HL-Net/blob/main/maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py) in PredCls and SGCls to filter the predicted relation predicates, which encourages diverse prediction results. 

## Help

Be free to contact me (`guoqing.zhang@bjtu.edu.cn`) if you have any questions!

## Acknowledgement

The code is implemented based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [PE-Net](https://github.com/VL-Group/PENET).
