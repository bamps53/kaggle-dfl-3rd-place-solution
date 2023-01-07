from copy import deepcopy

from datasets import video_transforms

from .pretrain import cfg

FPS = 25.0
HEIGHT, WIDTH = 360, 640

cfg = deepcopy(cfg)
cfg.exp_name = 'pretrain_b3_dwc'
cfg.output_dir = f'output/{cfg.exp_name}'
cfg.debug = False

cfg.train.df_path = '../input/SoccerNet/tracking/folds_resized360.csv'
cfg.train.data_dir = '../input/SoccerNet/tracking'
cfg.train.folder_name = 'img_resized360'

cfg.train.image_size = (HEIGHT, WIDTH)
cfg.train.transforms = video_transforms.Compose([
    video_transforms.RandomHorizontalFlip(),
    video_transforms.RandomRotation(5),
    video_transforms.ColorJitter(brightness=0.2, contrast=0.1),
    video_transforms.RandomCrop((int(HEIGHT*0.8), int(WIDTH*0.8))),
    video_transforms.Resize((HEIGHT, WIDTH)),
])
cfg.train.original_image_size = (360, 640)

cfg.valid.df_path = '../input/SoccerNet/tracking/folds_resized360.csv'
cfg.valid.data_dir = '../input/SoccerNet/tracking'
cfg.valid.folder_name = 'img_resized360'
cfg.valid.image_size = (HEIGHT, WIDTH)
cfg.valid.original_image_size = (360, 640)

cfg.model.model_name = 'tf_efficientnet_b3_ns'
cfg.model.in_channels = 1408
cfg.model.cls_loss_type = 'centernet'
cfg.model.temporal_shift = True
cfg.model.temporal_shift_dwc = True
cfg.model.drop = 0.3
cfg.model.drop_path = 0.2
cfg.model.grad_checkpointing = True

# others
cfg.lr = 1.0e-4
cfg.wd = 1.0e-3
cfg.min_lr = 1.0e-4
cfg.warmup_lr = 1.0e-4
cfg.warmup_epochs = 3
cfg.warmup = 1
cfg.epochs = 30
cfg.eval_intervals = 1
cfg.ema_start_epoch = 3
cfg.steps_per_epoch = 5000
