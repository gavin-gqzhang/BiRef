export CUDA_LAUNCH_BLOCKING=1

cuda_device=0,1,2,3
first_cuda=$(echo "$cuda_device" | cut -d ',' -f 1)
IFS=',' read -r -a array <<< "$cuda_device"
NUM_GUP=${#array[@]}

MODEL_NAME="BiRef"

GLOVE_DIR="/data/sdb/pretrain_ckpt/glove"
PRETRAIN_PATH='/data/sdb/pretrain_ckpt/pretrained_faster_rcnn'
DATA_DIR="/data/sdc/SGG_data"

USE_GT_BOX=False
USE_GT_OBJECT_LABEL=False
PREDICT_USE_BIAS=True

DATASET_CHOICE="GQA"
if [ "$DATASET_CHOICE" = "VG" ]; then
    SKIP_TEST=""
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/model_final.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
elif [ "$DATASET_CHOICE" = "GQA" ]; then
    SKIP_TEST=""
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1xGQA.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/gqa_model_final_from_vg.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
elif [ "$DATASET_CHOICE" = "OI_V4" ]; then
    SKIP_TEST="--skip-test"
    USE_GT_BOX=False
    USE_GT_OBJECT_LABEL=False
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x_for_OIV4.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/oiv4_det.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
elif [ "$DATASET_CHOICE" = "OI_V6" ]; then
    SKIP_TEST="--skip-test"
    USE_GT_BOX=False
    USE_GT_OBJECT_LABEL=False
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x_for_OIV6.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/oiv6_det.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
else
    echo "DATASET_CHOICE ValueError, must be 'VG', 'GQA', 'OI_V4', 'OI_V6'. "
    exit 1
fi

if [ "$USE_GT_BOX" = "True" ] && [ "$USE_GT_OBJECT_LABEL" = "True" ]; then
    mode="predcls"
elif [ "$USE_GT_BOX" = "True" ] && [ "$USE_GT_OBJECT_LABEL" = "False" ]; then
    mode="sgcls"
elif [ "$USE_GT_BOX" = "False" ] && [ "$USE_GT_OBJECT_LABEL" = "False" ]; then
    mode="sgdet"
else
    echo "Invalid combination of USE_GT_BOX and USE_GT_OBJECT_LABEL"
    exit 1
fi

if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
cp maskrcnn_benchmark/modeling/roi_heads/relation_head/model_utils.py $OUTPUT_DIR
cp maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py $OUTPUT_DIR


CUDA_VISIBLE_DEVICES=$cuda_device python -m torch.distributed.launch --nproc_per_node=$NUM_GUP --master_addr="127.0.0.1" --master_port=1642 tools/relation_train_net.py \
  --config-file $CONFIG_FILE $SKIP_TEST \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX $USE_GT_BOX \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL $USE_GT_OBJECT_LABEL \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS $PREDICT_USE_BIAS \
  MODEL.ROI_RELATION_HEAD.PREDICTOR $MODEL_NAME \
  MODEL.ROI_RELATION_HEAD.USE_PCR $USE_PCR \
  MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM $CONTEXT_HIDDEN_DIM \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH $(expr $NUM_GUP \* $PER_BATCH_SIZE) TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER $MAX_ITER SOLVER.BASE_LR $BASE_LR \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  SOLVER.PRE_VAL False \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 20000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  SOLVER.ACCUMULATE_GRAD $ACCUMULATE_GRAD \
  MODEL.PRETRAINED_DETECTOR_CKPT $PRETRAINED_DETECTOR_CKPT \
  SOLVER.DATASET_CHOICE $DATASET_CHOICE \
  DATASETS.DATA_DIR $DATA_DIR \
  GLOVE_DIR $GLOVE_DIR \
  OUTPUT_DIR $OUTPUT_DIR \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  TEST.ALLOW_LOAD_FROM_CACHE False \
  MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER 3 \
  ${@:1} ;

