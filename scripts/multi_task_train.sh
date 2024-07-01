export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

POSSIBLE_PATHS=(
    "$HOME/anaconda3"
    "/opt/anaconda3"
)

# 搜索并 source conda.sh
for path in "${POSSIBLE_PATHS[@]}"; do
    if [[ -f "$path/etc/profile.d/conda.sh" ]]; then
        source "$path/etc/profile.d/conda.sh"
        echo "Conda environment sourced from $path"
        break
    fi
done

conda activate sgg_benchmark

target_free_memory=20000
while true; do
    # 仅获取第一个GPU的显存总量和已使用量
    memory_info=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 2)
    
    # 计算空余显存
    total_memory=$(echo $memory_info | cut -d ',' -f 1 | tr -d '[:space:]')
    used_memory=$(echo $memory_info | cut -d ',' -f 2 | tr -d '[:space:]')
    free_memory=$((total_memory - used_memory))

    # 检查空余显存是否达到目标
    if [ "$free_memory" -gt "$target_free_memory" ]; then
        break
    else
        sleep 120
    fi
done

export CUDA_LAUNCH_BLOCKING=1

cuda_device=0,1,2,3
IFS=',' read -r -a array <<< "$cuda_device"
NUM_GUP=${#array[@]}

echo '************************* first task start *************************'

PER_BATCH_SIZE=2  # if PER_BATCH_SIZE=1 ==> BATCH_SIZE=4 ==> SOLVER.MAX_ITER=60000*2
MAX_ITER=80000   # if PER_BATCH_SIZE=2 ==> BATCH_SIZE=8 ==> SOLVER.MAX_ITER=60000
MODEL_NAME='EntityTrans_v3'

PRETRAINED_DETECTOR_CKPT="/data/sdc/pretrain_model/pretrained_faster_rcnn/model_final.pth"  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
GLOVE_DIR="/data/sdc/pretrain_model/glove"
ZEROSHOT_TYPE="None"

CUDA_VISIBLE_DEVICES=$cuda_device python -m torch.distributed.launch --nproc_per_node=$NUM_GUP --master_addr="127.0.0.1" --master_port=1642 tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR $MODEL_NAME \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH $(expr $NUM_GUP \* $PER_BATCH_SIZE) TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER $MAX_ITER SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  SOLVER.PRE_VAL False \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD $MAX_ITER \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  MODEL.PRETRAINED_DETECTOR_CKPT $PRETRAINED_DETECTOR_CKPT \
  GLOVE_DIR $GLOVE_DIR \
  OUTPUT_DIR outputs/${MODEL_NAME}_predcls_ada_reweight_with_bias \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  TEST.ALLOW_LOAD_FROM_CACHE False \
  SOLVER.ZEROSHOT_MODE $ZEROSHOT_TYPE \
  MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER 3 \
  ${@:1} \


# ************************** second task **************************
while true; do
    # 仅获取第一个GPU的显存总量和已使用量
    memory_info=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 2)
    
    # 计算空余显存
    total_memory=$(echo $memory_info | cut -d ',' -f 1 | tr -d '[:space:]')
    used_memory=$(echo $memory_info | cut -d ',' -f 2 | tr -d '[:space:]')
    free_memory=$((total_memory - used_memory))

    # 检查空余显存是否达到目标
    if [ "$free_memory" -gt "$target_free_memory" ]; then
        break
    else
        sleep 120
    fi
done

echo '************************* second task start *************************'

PER_BATCH_SIZE=2  # if PER_BATCH_SIZE=1 ==> BATCH_SIZE=4 ==> SOLVER.MAX_ITER=60000*2
MAX_ITER=80000   # if PER_BATCH_SIZE=2 ==> BATCH_SIZE=8 ==> SOLVER.MAX_ITER=60000
MODEL_NAME='EntityTrans_v3'

PRETRAINED_DETECTOR_CKPT="/data/sdc/pretrain_model/pretrained_faster_rcnn/model_final.pth"  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
GLOVE_DIR="/data/sdc/pretrain_model/glove"
ZEROSHOT_TYPE="None"

CUDA_VISIBLE_DEVICES=$cuda_device python -m torch.distributed.launch --nproc_per_node=$NUM_GUP --master_addr="127.0.0.1" --master_port=1672 tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR $MODEL_NAME \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH $(expr $NUM_GUP \* $PER_BATCH_SIZE) TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER $MAX_ITER SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  SOLVER.PRE_VAL False \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD $MAX_ITER \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  MODEL.PRETRAINED_DETECTOR_CKPT $PRETRAINED_DETECTOR_CKPT \
  GLOVE_DIR $GLOVE_DIR \
  OUTPUT_DIR outputs/${MODEL_NAME}_sgcls_ada_reweight_with_bias \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  TEST.ALLOW_LOAD_FROM_CACHE False \
  SOLVER.ZEROSHOT_MODE $ZEROSHOT_TYPE \
  MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER 3 \
  ${@:1} \
