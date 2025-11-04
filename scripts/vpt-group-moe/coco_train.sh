# !/bin/bash



# vit_small_dinov2 coco14 224
for worknum in 1
do
    torchrun --nnodes=1 --nproc_per_node=4 --standalone --rdzv_id=100 --rdzv_backend=c10d \
    main_mlic.py --cfg config/vpt-group-moe-g5e3/vit_small_dinov2/coco14_224.yaml \
    --output checkpoint/vpt-group-moe-g5e3/code/vit_small_dinov2/224/coco14/seed42/worknum_${worknum} \
    --gpus 0,1,2,3 \
    --seed 42 \
    --print-freq 400
done