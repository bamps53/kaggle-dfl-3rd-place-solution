from copy import deepcopy
from types import SimpleNamespace

cfg = SimpleNamespace(**{})
cfg = deepcopy(cfg)
cfg.project = 'kaggle-dfl-pt'
cfg.debug = False

cfg.train = SimpleNamespace(**{})
cfg.train.df_path = '../input/folds.csv'
cfg.train.all_df_path = '../input/folds_all.csv'
cfg.train.video_feature_dir = '../input/train_frames_quarter'
cfg.train.label_dir = '../input/event_labels'
cfg.train.duration = 32
cfg.train.offset = 5
cfg.train.batch_size = 4
cfg.train.num_workers = 4
cfg.train.bg_sampling_rate = 0.0
cfg.train.sigma = 3
cfg.train.transforms = None
cfg.train.down_ratio = 32
cfg.train.original_image_size = (1080, 1920)
cfg.train.transforms = None

cfg.valid = SimpleNamespace(**{})
cfg.valid.df_path = '../input/folds.csv'
cfg.valid.all_df_path = '../input/folds_all.csv'
cfg.valid.video_feature_dir = '../input/train_frames_quarter'
cfg.valid.label_dir = '../input/event_labels'
cfg.valid.duration = 32
cfg.valid.offset = 5
cfg.valid.batch_size = 4
cfg.valid.num_workers = 4
cfg.valid.transforms = None
cfg.valid.down_ratio = 32
cfg.valid.original_image_size = (1080, 1920)

cfg.model = SimpleNamespace(**{})
cfg.model.pretrained = True
cfg.model.model_name = 'tf_efficientnet_b0_ns'
cfg.model.in_channels = 1280
cfg.model.num_classes = 3
cfg.model.class_weight = None
cfg.model.init_bias = None
cfg.model.scale_factor = 2
cfg.model.cls_weight = 1.0
cfg.model.reg_weight = 0.2
cfg.model.pretrained_path = None
cfg.model.cls_loss_type = 'bce'
cfg.model.norm_type = 'ln'
cfg.model.duration = 32
cfg.model.drop_rate = 0.0
cfg.model.freeze_backbone = False
cfg.model.gamma = 2.0
cfg.model.alpha = 0.25
cfg.model.beta = 0.9999
cfg.model.temporal_shift = False
cfg.model.non_local = False
cfg.model.se = False
cfg.model.attention = True
cfg.model.drop = 0.0
cfg.model.drop_path = 0.0
cfg.model.drop_block = 0.0
cfg.model.grad_checkpointing = False

# others
cfg.seed = 42
cfg.device = 'cuda'
cfg.lr = 1.0e-3
cfg.warmup = 1
cfg.epochs = 100
cfg.eval_intervals = 1
cfg.mixed_precision = True
cfg.resume = False
cfg.ema_start_epoch = 3
