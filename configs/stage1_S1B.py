from default_cv_config import basic_cv_cfg as cfg
import os
import albumentations as A


# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.train_df = cfg.data_dir + "meta_wirbel_dcm_v2.csv"
cfg.output_dir = f"./output/models/{os.path.basename(__file__).split('.')[0]}"
cfg.mask_folder = cfg.data_dir + "masks_2d/"

# stages
cfg.train = True

# model
cfg.model = "stage1_ch_mdl_2"
cfg.backbone = "tf_efficientnet_b3_ns"
cfg.drop_out = 0.0
cfg.mixup_probability = 1.0
cfg.mixadd = False
cfg.mix_beta = 1.0
cfg.return_logits = False

# DATASET
cfg.dataset = "stage1_ch_ds_2b"
cfg.classes = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
cfg.image_width = 320
cfg.image_height = 320
cfg.normalization = "simple"


# OPTIMIZATION & SCHEDULE
cfg.fold = -1
cfg.seed = 42
cfg.epochs = 40
cfg.lr = 0.0005

cfg.optimizer = "AdamW"
cfg.weight_decay = 0.0
cfg.warmup = 0.0
cfg.batch_size = 64
cfg.num_workers = 8


# Postprocess
cfg.post_process_pipeline = "stage1_pp_pp_1"
cfg.stage = 1

cfg.train_aug = A.Compose(
    [
        A.Resize(int(cfg.image_height * 1.125), int(cfg.image_width * 1.125)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=25, p=0.5),
        A.RandomCrop(
            always_apply=False, p=1.0, height=cfg.image_height, width=cfg.image_width
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5
        ),
        A.Cutout(num_holes=8, max_h_size=36, max_w_size=36, p=0.8),
    ]
)


cfg.val_aug = A.Compose(
    [
        A.Resize(int(cfg.image_height * 1.125), int(cfg.image_width * 1.125)),
        A.CenterCrop(
            always_apply=False, p=1.0, height=cfg.image_height, width=cfg.image_width
        ),
    ]
)
