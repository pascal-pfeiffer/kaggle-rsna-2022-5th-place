from types import SimpleNamespace


cfg = SimpleNamespace(**{})

# stages
cfg.train = True
cfg.test = False

# dataset
cfg.dataset = "ds_dummy"
cfg.batch_size = 32
cfg.normalization = None
cfg.train_aug = None
cfg.val_aug = None

# training routine
cfg.fold = 0
cfg.val_fold = -1
cfg.lr = 1e-4
cfg.schedule = "cosine"
cfg.num_cycles = 0.5
cfg.weight_decay = 0
cfg.optimizer = "AdamW"
cfg.epochs = 10
cfg.seed = -1
cfg.simple_eval = False
cfg.do_test = True
cfg.do_seg = False
cfg.eval_ddp = True
cfg.debug = False
cfg.save_val_data = True
cfg.gradient_checkpointing = False

# eval
cfg.post_process_pipeline = "pp_dummy"

# ressources
cfg.mixed_precision = True
cfg.grad_accumulation = 1
cfg.syncbn = False
cfg.gpu = 0
cfg.num_workers = 4
cfg.drop_last = True
cfg.save_checkpoint = True
cfg.save_only_last_ckpt = False
cfg.save_weights_only = True
cfg.pin_memory = False

# logging,
cfg.tags = None
cfg.sgd_nesterov = True
cfg.sgd_momentum = 0.9
cfg.loss = "bce"

# dirs
cfg.data_dir = "./input/rsna-2022-cervical-spine-fracture-detection/"
cfg.data_folder = cfg.data_dir + "train_images/"
cfg.test_data_folder = cfg.data_dir + "train_images/"

basic_cfg = cfg
