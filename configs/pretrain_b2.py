from copy import deepcopy

from datasets import video_transforms 

from .pretrain import cfg

FPS = 25.0
HEIGHT, WIDTH = 288, 512

cfg = deepcopy(cfg)
cfg.project = 'kaggle-dfl-pt'
cfg.exp_name = 'stage3_ball022_b2_aug_clip_no_warmup_0918_sn'
cfg.output_dir = f'output/{cfg.exp_name}'
cfg.debug = False

cfg.train.df_path = '../input/SoccerNet/tracking/folds_resized288.csv'
cfg.train.data_dir = '../input/SoccerNet/tracking288'
cfg.train.folder_name = 'img_resized288'
cfg.train.duration = 32
cfg.train.batch_size = 4
cfg.train.num_workers = 4 if not cfg.debug else 0
cfg.train.image_size = (HEIGHT, WIDTH)
cfg.train.down_ratio = 32
cfg.train.transforms = video_transforms.Compose([
    video_transforms.RandomHorizontalFlip(),
    video_transforms.RandomRotation(5),
    video_transforms.ColorJitter(brightness=0.2, contrast=0.1),
    video_transforms.RandomCrop((int(HEIGHT*0.8), int(WIDTH*0.8))),
    video_transforms.Resize((HEIGHT, WIDTH)),
])
cfg.train.original_image_size = (288, 512)

cfg.valid.df_path = '../input/SoccerNet/tracking/folds_resized288.csv'
cfg.valid.data_dir = '../input/SoccerNet/tracking288'
cfg.valid.folder_name = 'img_resized288'
cfg.valid.duration = 32
cfg.valid.offset = 5
cfg.valid.batch_size = 4
cfg.valid.num_workers = 4 if not cfg.debug else 0
cfg.valid.image_size = (HEIGHT, WIDTH)
cfg.valid.down_ratio = 32
cfg.valid.original_image_size = (288, 512)

# cfg.test.video_paths = sorted(glob.glob('../input/dfl-bundesliga-data-shootout/test/*'))
# if len(cfg.test.video_paths) == 32: # public test
#     cfg.test.video_paths = cfg.test.video_paths[:1]
cfg.test.video_paths = [
    '../input/dfl-bundesliga-data-shootout/train/9a97dae4_1.mp4',
    '../input/dfl-bundesliga-data-shootout/train/ecf251d4_0.mp4']
cfg.test.duration = 32
cfg.test.offset = 5
cfg.test.batch_size = 32
cfg.test.num_workers = 0
cfg.test.score_th = 0.01
cfg.test.nms_thresholds = [12, 6, 6]
cfg.test.weight_paths = [
    '../input/stage0-021-fold0/best_fold0.pth',
    '../input/stage0-016-fold0/best_fold0.pth']
cfg.test.image_size = (HEIGHT, WIDTH)

cfg.model.model_name = 'tf_efficientnet_b2_ns'
cfg.model.in_channels = 1408
cfg.model.num_classes = 3
cfg.model.cls_weight = 1.0
cfg.model.reg_weight = 0.2
cfg.model.cls_loss_type = 'centernet'
cfg.model.norm_type = 'ln'
cfg.model.duration = 32
cfg.model.pretrained_path = None
cfg.model.alpha = 0.25
cfg.model.beta = 0.9999
cfg.model.temporal_shift = True
cfg.model.temporal_shift_dwc = False
cfg.model.drop = 0.3
cfg.model.drop_path = 0.2
cfg.model.drop_block = 0.0
cfg.model.grad_checkpointing = True

# others
cfg.seed = 42
cfg.device = 'cuda'
cfg.lr = 1.0e-4
cfg.wd = 1.0e-3
cfg.min_lr = 1.0e-4
cfg.warmup_lr = 1.0e-4
cfg.warmup_epochs = 3
cfg.warmup = 1
cfg.epochs = 80
cfg.eval_intervals = 1
cfg.mixed_precision = True
cfg.ema_start_epoch = 3
cfg.steps_per_epoch = 2000