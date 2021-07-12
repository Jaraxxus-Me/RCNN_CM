export CUDA_VISIBLE_DEVICES=0
# number of shots
for j in 1 2 3 4
do
# few-shot fine-tuning
python finetune_baseline.py --data_path "/home/user/ws/dataset/coco" \
--resume "./baseline_weights/mobile-base-24.pth" \
--meta_type $j --shots 10
done
