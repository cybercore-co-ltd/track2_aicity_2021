python tools/train.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnext101_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/cybercore/su/AICity2021-VOC-ReID/resnext101_ibn_a.pth.tar')" \
SOLVER.LR_SCHEDULER 'cosine_step' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'softmax' \
MODEL.METRIC_LOSS_TYPE 'none' \
DATALOADER.SAMPLER 'softmax' \
INPUT.PROB 0.0 \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
MODEL.IF_LABELSMOOTH 'on' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 12 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
SOLVER.IMS_PER_BATCH 32 \
DATASETS.TRAIN "('aicity20-ReOri',)" \
DATASETS.TEST "('aicity20-ReOri',)" \
DATASETS.ROOT_DIR "('/media/data/ai-city/Track2/AIC21_Track2_ReID_Simulation/')" \
OUTPUT_DIR "('./output/aicity21_ori')"