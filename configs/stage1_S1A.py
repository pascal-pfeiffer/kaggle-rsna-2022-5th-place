from default_cv_config import basic_cv_cfg as cfg
import os


# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.train_df = cfg.data_dir + "meta_wirbel_dcm_v2.csv"
cfg.test_df = cfg.data_dir + "test_meta_wirbel_dcm_v2.csv"
cfg.output_dir = f"./output/models/{os.path.basename(__file__).split('.')[0]}"

# stages
cfg.train = True
cfg.test = True

# model
cfg.model = "stage1_pp_mdl_1"
cfg.backbone = "tf_efficientnet_b5_ns"
cfg.drop_out = 0.0

# DATASET
cfg.dataset = "stage1_pp_ds_1"
cfg.classes = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
cfg.image_width = 512
cfg.image_height = 512
cfg.normalization = "simple"

# OPTIMIZATION & SCHEDULE
cfg.fold = -1
cfg.seed = 42
cfg.epochs = 3
cfg.lr = 0.0005

cfg.optimizer = "AdamW"
cfg.weight_decay = 0.0
cfg.warmup = 0.0
cfg.batch_size = 8
cfg.num_workers = 8

# Postprocess
cfg.post_process_pipeline = "stage1_pp_pp_1"
cfg.stage = 1
