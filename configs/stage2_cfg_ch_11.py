from default_cv_config import basic_cv_cfg as cfg
import os
import albumentations as A
import cv2

# paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.train_df = cfg.data_dir + "labels_fracture_2D_v1_ff.csv"
cfg.output_dir = f"./output/models/{os.path.basename(__file__).split('.')[0]}"
cfg.box_3d = cfg.data_dir + "crop_box_3d.csv"

# stages
cfg.train = True


cfg.cache_size = 200
cfg.cache_disk = ""


# model
cfg.model = "stage2_ps_mdl_3b"
cfg.backbone = "efficientnetv2_rw_t"
cfg.drop_out = 0.5
cfg.mixup_probability = 1.0
cfg.mixadd = True
cfg.mix_beta = 1.0
cfg.use_3d_centerpooling = False
cfg.pool = "avg"
cfg.aux_loss_weight = 0.25
cfg.loss = "bce"  # weighted_bce
cfg.num_3d_layers = 0
cfg.sample_weights = True
cfg.sample_weights_global = False

# DATASET
cfg.dataset = "stage2_ch_ds_2"
cfg.classes = [
    "label_frac_c1",
    "label_frac_c2",
    "label_frac_c3",
    "label_frac_c4",
    "label_frac_c5",
    "label_frac_c6",
    "label_frac_c7",
]
cfg.aux_classes = [
    "label_vert_c1",
    "label_vert_c2",
    "label_vert_c3",
    "label_vert_c4",
    "label_vert_c5",
    "label_vert_c6",
    "label_vert_c7",
]

cfg.image_width = 320
cfg.image_height = 320
cfg.image_width_loading = 512  # int(cfg.image_width * 1.125)
cfg.image_height_loading = 512  # int(cfg.image_height * 1.125)
cfg.normalization = "simple"
cfg.frames_num = 1
cfg.frames_step_size = 5
cfg.stack_neighbor_frames = 1
cfg.image_channels = cfg.stack_neighbor_frames * 2 + 1


# OPTIMIZATION & SCHEDULE
cfg.fold = -1
cfg.seed = 42
cfg.epochs = 2
cfg.lr = 0.0005

cfg.optimizer = "AdamW"
cfg.weight_decay = 0.0
cfg.warmup = 0.0
cfg.batch_size = 64
cfg.num_workers = 8
cfg.num_workers_dataloader = 1


# Postprocess
cfg.post_process_pipeline = "stage2_pp_ps_1"
cfg.stage = 2

cfg.resize_aug = A.Compose(
    [
        A.LongestMaxSize(cfg.image_height_loading, p=1),
        A.PadIfNeeded(
            cfg.image_height_loading,
            cfg.image_width_loading,
            border_mode=cv2.BORDER_CONSTANT,
            p=1,
        ),
    ],
    p=1.0,
)

cfg.train_aug = A.Compose(
    [
        A.LongestMaxSize(360, p=1),
        A.PadIfNeeded(
            cfg.image_height_loading,
            cfg.image_width_loading,
            border_mode=cv2.BORDER_CONSTANT,
            p=1,
        ),
        A.HorizontalFlip(always_apply=False, p=0.5),
        A.VerticalFlip(always_apply=False, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25), contrast_limit=(-0.25, 0.25), p=1.0
        ),
        A.ShiftScaleRotate(
            always_apply=False,
            p=1.0,
            shift_limit_x=(-0.1, 0.1),
            shift_limit_y=(-0.1, 0.1),
            scale_limit=(-0.25, 0.25),
            rotate_limit=(-25, 25),
            interpolation=1,
            border_mode=4,
            value=None,
            mask_value=None,
            # rotate_method="largest_box",
        ),
        A.RandomCrop(
            always_apply=False, p=1.0, height=cfg.image_height, width=cfg.image_width
        ),
        A.CoarseDropout(
            always_apply=False,
            p=0.5,
            max_holes=8,
            max_height=0.2,
            max_width=0.2,
            min_holes=2,
            min_height=0.0625,
            min_width=0.0625,
            fill_value=0,
            mask_fill_value=None,
        ),
    ],
    p=1.0,
    bbox_params=None,
    keypoint_params=None,
    additional_targets={},
)

cfg.val_aug = A.Compose(
    [
        A.LongestMaxSize(360, p=1),
        A.PadIfNeeded(
            cfg.image_height_loading,
            cfg.image_width_loading,
            border_mode=cv2.BORDER_CONSTANT,
            p=1,
        ),
        A.CenterCrop(
            always_apply=False, p=1.0, height=cfg.image_height, width=cfg.image_width
        ),
    ]
)
