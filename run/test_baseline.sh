export CUDA_VISIBLE_DEVICES=3
# number of shots
for j in 1 2 3 4 5 6
do
# testing on base and novel class
python test_baseline.py --data_path "/home/user/ws/dataset/coco" \
--meta_type $j
done
