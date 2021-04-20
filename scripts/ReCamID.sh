python tools/train.py --config_file='configs/aicity20.yml' \
MODEL.DEVICE_ID "('1')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnext101_ibn_a')" \
MODEL.PRETRAIN_PATH "('/home/cybercore/su/AICity2021-VOC-ReID/resnext101_ibn_a.pth.tar')" \
SOLVER.LR_SCHEDULER 'cosine_step' \
DATALOADER.NUM_INSTANCE 8 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 12 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
SOLVER.IMS_PER_BATCH 32 \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
INPUT.SIZE_TRAIN '([320, 320])' \
INPUT.SIZE_TEST '([320, 320])' \
DATASETS.TRAIN "('aicity20-ReCam',)" \
DATASETS.TEST "('veri',)" \
DATASETS.ROOT_DIR "('/media/data/ai-city/Track2')" \
OUTPUT_DIR "('./output/aicity20/0409-ensemble/ReCamID')"