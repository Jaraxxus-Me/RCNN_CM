export CUDA_VISIBLE_DEVICES=0
python train_find.py --bs 2 --bs_v 2 --cls 0.5 --data_path "/home/user/ws/FSDet/data" --output_dir "./save_find_0.5" --epochs 20
