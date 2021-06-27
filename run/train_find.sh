export CUDA_VISIBLE_DEVICES=0
python train_find.py --bs 2 --bs_v 2 --cls 0.2 --data_path "/home/user/ws/FSDet/data" --epochs 20
