export CUDA_VISIBLE_DEVICES=1
# number of shots
for j in 1 2 3 5 10
do
# few-shot fine-tuning
python finetune_baseline.py --data_path "/home/user/ws/FSDet/data" \
--epochs 10 --bs 4 --bs_v 2 \
--resume "./baseline_weights_res/resnet101-baseline-21.pth" \
--meta_type 1 --shots $j
done
