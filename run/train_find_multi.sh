export CUDA_VISIBLE_DEVICES=0,2
python -m torch.distributed.launch --nproc_per_node=2 --use_env train_find_multi.py --data_path "/home/user/ws/FSDet/data"