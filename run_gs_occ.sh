# train nuscenes
CUDA_VISIBLE_DEVICES=0,1,2,3
config=configs/nusc-sem-gs.txt

ckpts=ckpts/stage1_pose_nusc
torchrun --nproc_per_node=1 run.py --config $config \
# torchrun --nproc_per_node=1 run.py --h \
# python -m torch.distributed.launch --nproc_per_node=8 run.py --config $config
--load_weights_folder $ckpts \
# --eval_only 

# eval nuscenes
# ckpts='ckpts/nusc-sem-gs'
# python -m torch.distributed.launch --nproc_per_node=8 run.py --config $config \
# --load_weights_folder $ckpts \
# --eval_only 


#############-----DDAD-----#####################

# # train ddad 
# config=configs/ddad-sem-gs.txt

# ckpts='ckpts/stage1_pose_ddad'
# python -m torch.distributed.launch --nproc_per_node=8 run.py --config $config \
# --load_weights_folder $ckpts \
# # --eval_only 

# # eval ddad
# ckpts='ckpts/ddad-sem-gs'
# python -m torch.distributed.launch --nproc_per_node=8 run.py --config $config \
# --load_weights_folder $ckpts \
# --eval_only 

# sh /home/wsgan/project/bev/GaussianOcc/run_gs_occ.sh