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
cfg.train.duration = 16
cfg.train.offset = 5
cfg.train.batch_size = 4
cfg.train.num_workers = 0
cfg.train.bg_sampling_rate = 0.0
cfg.train.sigma = 3
cfg.train.transforms = None
cfg.train.crop_roi = False
cfg.train.roi_path = '../input/roi_dict.json'
cfg.train.crop_margin = 10
cfg.train.crop_min_size = (288, 512)
cfg.train.crop_noise = 0.0
cfg.train.original_shape = 540, 960

cfg.valid = SimpleNamespace(**{})
cfg.valid.df_path = '../input/folds.csv'
cfg.valid.all_df_path = '../input/folds_all.csv'
cfg.valid.video_feature_dir = '../input/train_frames_quarter'
cfg.valid.label_dir = '../input/event_labels'
cfg.valid.duration = 16
cfg.valid.offset = 5
cfg.valid.batch_size = 4
cfg.valid.num_workers = 0
cfg.valid.transforms = None
cfg.valid.crop_roi = False
cfg.valid.roi_path = '../input/roi_dict.json'
cfg.valid.crop_margin = 10
cfg.valid.crop_min_size = (288, 512)
cfg.valid.crop_noise = 0.0
cfg.valid.original_shape = 540, 960

cfg.test = SimpleNamespace(**{})
cfg.test.video_feature_dir = 'train_frames_8'
cfg.test.duration = 16
cfg.test.offset = 5
cfg.test.batch_size = 4
cfg.test.num_workers = 0
cfg.test.score_th = 0.05
cfg.test.nms_th = 10
cfg.test.weight_path = '../input/pt024-fold0/best_fold0.pth'

cfg.model = SimpleNamespace(**{})
cfg.model.pretrained = True
cfg.model.model_name = 'tf_efficientnet_b0_ns'
cfg.model.in_channels = 1280
cfg.model.num_classes = 3
cfg.model.class_weight = None
cfg.model.init_bias = None
cfg.model.scale_factor = 2
cfg.model.cls_weight = 1.0
cfg.model.reg_weight = 0.1
cfg.model.pretrained_path = None
cfg.model.cls_loss_type = 'bce'
cfg.model.norm_type = 'ln'
cfg.model.duration = 32
cfg.model.drop_rate = 0.0
cfg.model.freeze_backbone = False
cfg.model.gamma = 2.0
cfg.model.alpha = 0.25
cfg.model.temporal_shift = False
cfg.model.temporal_shift_dwc = False
cfg.model.non_local = False
cfg.model.se = False
cfg.model.res = False
cfg.model.n_div = 8
cfg.model.mix_beta = 0.0
cfg.model.manifold_mixup = False
cfg.model.both_mixup = False
cfg.model.drop = 0.0
cfg.model.drop_path = 0.0
cfg.model.drop_block = 0.0
cfg.model.manual_weights = None
cfg.model.resume_exp = None
cfg.model.rescale_layer_norm = False
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
cfg.ema_start_epoch = -1
