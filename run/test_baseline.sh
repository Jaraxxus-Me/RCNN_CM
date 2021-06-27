export CUDA_VISIBLE_DEVICES=0
# number of shots
for j in 1 2 3 5 10
do
# testing on base and novel class
python test_baseline.py --data_path "/home/user/ws/FSDet/data" \
--resume "./fine_baseline_weight" --bs 4 --shots $j
done
