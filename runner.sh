config=configs/nusc-sem-gs.txt
ckpts='ckpts/nusc-sem-gs'
torchrun  --nproc_per_node=1 runner.py --config $config \
--load_weights_folder $ckpts \
--eval_only 