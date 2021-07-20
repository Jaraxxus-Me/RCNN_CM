export CUDA_VISIBLE_DEVICES=0
# number of shots
for j in 1 2 3 4 5 6
do
# few-shot fine-tuning
python finetune_find.py --data_path "/home/user/ws/dataset/coco" \
--meta_type $j --shots 1 --dataset "coco"
done
