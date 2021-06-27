export CUDA_VISIBLE_DEVICES=2
python train_mobilenetv2_meta.py --bs 4 --bs_v 4 --cls 0.1 --data_path "/home/user/ws/FSDet/data" --output_dir "./save_model_meta_0.1"
