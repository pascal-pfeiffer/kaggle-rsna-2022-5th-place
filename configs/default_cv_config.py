from default_config import basic_cfg

cfg = basic_cfg

# img model
cfg.backbone = "tf_efficientnet_b0_ns"
cfg.pretrained = True
cfg.pool = "avg"
cfg.in_chans = 3
cfg.gem_p_trainable = False
cfg.dropout = 0.0
cfg.warmup = 0.0

basic_cv_cfg = cfg
