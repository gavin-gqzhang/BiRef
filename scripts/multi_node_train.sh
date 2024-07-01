export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export NCCL_SOCKET_IFNAME="eno1"

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

conda activate maskrcnn

export CUDA_LAUNCH_BLOCKING=1

target_free_memory=20000
cuda_device=0,1,2,3
first_cuda=$(echo "$cuda_device" | cut -d ',' -f 1)
IFS=',' read -r -a array <<< "$cuda_device"
NUM_GUP=${#array[@]}

while true; do
    # 仅获取第一个GPU的显存总量和已使用量
    memory_info=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$first_cuda")
    
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

MASTER_ADDR="10.126.62.187"
NUM_NODE=2
NODE_RANK=0  # if master node, NODE_RANK=0

PER_BATCH_SIZE=2  # if PER_BATCH_SIZE=1 ==> BATCH_SIZE=4 ==> SOLVER.MAX_ITER=60000*2
MAX_ITER=40000   # if PER_BATCH_SIZE=2 ==> BATCH_SIZE=8 ==> SOLVER.MAX_ITER=60000
MODEL_NAME='EntityTrans_v3'

GLOVE_DIR="/data/sdb/pretrain_ckpt/glove"
PRETRAIN_PATH='/data/sdb/pretrain_ckpt/pretrained_faster_rcnn'
DATA_DIR="/data/sdb/SGG_data"

OUTPUT_DIR=outputs/$DATASET_CHOICE/${MODEL_NAME}_sgcls_withbias

DATASET_CHOICE="GQA"
if [ "$DATASET_CHOICE" = "VG" ]; then
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/model_final.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
elif [ "$DATASET_CHOICE" = "GQA" ]; then
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1xGQA.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/gqa_model_final_from_vg.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
elif [ "$DATASET_CHOICE" = "OI_V4" ]; then
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x_for_OIV4.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/oiv4_det.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
elif [ "$DATASET_CHOICE" = "OI_V6" ]; then
    CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x_for_OIV6.yaml"
    PRETRAINED_DETECTOR_CKPT=$PRETRAIN_PATH/oiv6_det.pth  # "/data/sdb/pretrain_ckpt/pretrained_faster_rcnn/model_final.pth"
else
    echo "DATASET_CHOICE ValueError, must be 'VG', 'GQA', 'OI_V4', 'OI_V6'. "
    exit 1
fi

if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
cp maskrcnn_benchmark/modeling/roi_heads/relation_head/model_utils.py $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$cuda_device python -m torch.distributed.launch --nnodes $NUM_NODE --nproc_per_node $NUM_GUP --master_addr $MASTER_ADDR --master_port 1642 --node_rank $NODE_RANK tools/relation_train_net.py \
  --config-file $CONFIG_FILE \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR $MODEL_NAME \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH $(expr $NUM_GUP \* $PER_BATCH_SIZE \* $NUM_NODE ) TEST.IMS_PER_BATCH $(expr $NUM_GUP \* $NUM_NODE ) \
  SOLVER.MAX_ITER $MAX_ITER SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  SOLVER.PRE_VAL False \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD $MAX_ITER \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  MODEL.PRETRAINED_DETECTOR_CKPT $PRETRAINED_DETECTOR_CKPT \
  SOLVER.DATASET_CHOICE $DATASET_CHOICE \
  DATASETS.DATA_DIR $DATA_DIR \
  GLOVE_DIR $GLOVE_DIR \
  OUTPUT_DIR $OUTPUT_DIR \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  TEST.ALLOW_LOAD_FROM_CACHE False \
  MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER 3 \
  ${@:1} ;