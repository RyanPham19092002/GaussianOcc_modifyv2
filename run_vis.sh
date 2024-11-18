# cd /home/wsgan/project/bev/GaussianOcc

# nuscene
config=configs/nusc-sem-gs-vis.txt
# ckpts='ckpts/nusc-sem-gs'
ckpts='/home/vinai/Workspace/phat-intern-dev/VinAI/GaussianOcc/logs/nusc-sem-gs_all/seg_False_0.02_0.1_0.1_pose_yes_yes_mask_False/type_3dgs_0.0_s_0.2_384_l_self_en_w_0.1_ep_8_f_False_infi_True_cont_True_depth_0.1_80.0_101/standard/models/weights_3233'
# torchrun  --nproc_per_node=1 run_vis.py --config $config \
torchrun  --nproc_per_node=1 run_vis_gs.py --config $config \
--load_weights_folder $ckpts \
--eval_only 

# ddad
# config=configs/ddad-sem-gs.txt
# ckpts='ckpts/ddad-sem-gs'
# python -m torch.distributed.launch --nproc_per_node=1 run_vis.py --config $config \
# --load_weights_folder $ckpts \
# --eval_only 

# sh /home/wsgan/project/bev/GaussianOcc/run_vis.sh
