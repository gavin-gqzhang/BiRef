
source /home/dell/anaconda3/etc/profile.d/conda.sh
conda activate sgg_benchmark

NUM_GUP=1
MODEL_NAME='Qformer_contrast'

python -m torch.distributed.launch --nproc_per_node=$NUM_GUP --master_addr="127.0.0.1" --master_port=1674 tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR $MODEL_NAME \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 30000 \
  SOLVER.CHECKPOINT_PERIOD 30000 \
  MODEL.PRETRAINED_DETECTOR_CKPT /data/sdc/pretrain_model/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR outputs/${MODEL_NAME}_withgate \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0;