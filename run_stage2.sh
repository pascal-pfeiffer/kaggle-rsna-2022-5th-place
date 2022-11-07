python scripts/get_labels_fracture_2D_v1_ff.py
python scripts/process_3d_box_mask.py

torchrun --nproc_per_node=3 train.py -C stage2_cfg_ps_wd_29_ff
torchrun --nproc_per_node=3 train.py -C stage2_cfg_ps_wd_30_ff
torchrun --nproc_per_node=3 train.py -C stage2_cfg_ps_wd_37_ff
torchrun --nproc_per_node=3 train.py -C stage2_cfg_ps_wd_40_ff

torchrun --nproc_per_node=3 train.py -C stage2_cfg_ch_11
