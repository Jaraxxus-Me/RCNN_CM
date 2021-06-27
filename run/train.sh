export CUDA_VISIBLE_DEVICES=1
python train_find.py --bs 2 --bs_v 2 --data_path "/home/user/ws/FSDet/data" --output_dir "./save_model_meta_0.3" --resume "./save_model_meta_0.3/mobile-model-19.pth" --epochs 20
