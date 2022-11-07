python scripts/get_reversed_dicoms.py
python scripts/create_train_folded_v1.py
python scripts/preprocess_mask_2d.py
python scripts/get_meta_wirbel_dcm_v1_and_v2.py

torchrun --nproc_per_node=5 train.py -C stage1_S1A
torchrun --nproc_per_node=1 train.py -C stage1_S1B
