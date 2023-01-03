from copy import deepcopy
import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from timm.scheduler import CosineLRScheduler

try:
    # import training only modules
    import wandb
except:
    print('wandb is not installed.')

from configs.base_e2e import cfg
from metrics.ap import event_detection_ap, tolerances
from models.unet import ImageUNetLNMixup as Net
from utils.debugger import set_debugger
from utils.common import set_seed, create_checkpoint, resume_checkpoint, batch_to_device,  nms, log_results
from utils.ema import ModelEmaV2
from datasets.e2e import get_trainval_dataloader, get_full_val_dataloader, get_video_dataloader, torch_pad_if_needed
try:
    from torchvideotransforms import video_transforms
except:
    video_transforms = None
    print('torchvideotransforms is not installed.')


EVENT_CLASSES = [
    'challenge',
    'play',
    'throwin'
]
FPS = 25.0
HEIGHT, WIDTH = 360, 640
DURATION = 64

cfg = deepcopy(cfg)
cfg.project = 'kaggle-dfl-pt'
cfg.exp_name = 'final_all_mix_360_b2_dur64_noreg_nomask'
cfg.output_dir = f'output/{cfg.exp_name}'
cfg.debug = False

cfg.train.df_path = '../input/folds.csv'
cfg.train.video_feature_dir = '../input/train_frames'
cfg.train.label_dir = '../input/event_labels'
cfg.train.duration = DURATION
cfg.train.offset = 5
cfg.train.batch_size = 4
cfg.train.num_workers = 8 if not cfg.debug else 0
cfg.train.image_size = (HEIGHT, WIDTH)
cfg.train.bg_sampling_rate = 0.5
cfg.train.transforms = video_transforms.Compose([
    video_transforms.RandomHorizontalFlip(),
    video_transforms.RandomRotation(10),
    video_transforms.ColorJitter(brightness=0.2, contrast=0.1),
    # video_transforms.RandomCrop((int(HEIGHT*0.8), int(WIDTH*0.8))),
    video_transforms.Resize((HEIGHT, WIDTH)),
])

cfg.valid.df_path = '../input/folds.csv'
cfg.valid.all_df_path = '../input/folds_all.csv'
cfg.valid.video_feature_dir = '../input/train_frames'
cfg.valid.label_dir = '../input/event_labels'
cfg.valid.duration = DURATION
cfg.valid.offset = 5
cfg.valid.batch_size = 4
cfg.valid.num_workers = 4 if not cfg.debug else 0
cfg.valid.image_size = (HEIGHT, WIDTH)

# cfg.test.video_paths = sorted(glob.glob('../input/dfl-bundesliga-data-shootout/test/*'))
# if len(cfg.test.video_paths) == 32: # public test
#     cfg.test.video_paths = cfg.test.video_paths[:1]
cfg.test.video_paths = [
    '../input/dfl-bundesliga-data-shootout/train/9a97dae4_1.mp4',
    '../input/dfl-bundesliga-data-shootout/train/ecf251d4_0.mp4']
cfg.test.duration = DURATION
cfg.test.offset = 5
cfg.test.batch_size = DURATION
cfg.test.num_workers = 0
cfg.test.score_th = 0.01
cfg.test.nms_thresholds = [12, 6, 6]
cfg.test.weight_paths = [
    '../input/stage0-021-fold0/best_fold0.pth',
    '../input/stage0-032-fold0/best_fold0.pth']
cfg.test.image_size = (HEIGHT, WIDTH)

cfg.model.model_name = 'tf_efficientnet_b2_ns'
cfg.model.in_channels = 1408
cfg.model.num_classes = 3
cfg.model.cls_weight = 1.0
cfg.model.reg_weight = 0.0
cfg.model.cls_loss_type = 'no_mask_cb_focal'
cfg.model.norm_type = 'ln'
cfg.model.duration = DURATION
cfg.model.pretrained_path = './output/stage3_ball022_b2_aug_clip_no_warmup_0918_sn/best_fold0.pth'
# cfg.model.resume_exp = 'stage3_ball022_b2_aug_clip_no_warmup_0918_sn'
cfg.model.alpha = 0.25
cfg.model.beta = 0.9999
cfg.model.temporal_shift = True
cfg.model.drop = 0.3
cfg.model.drop_path = 0.2
cfg.model.drop_block = 0.0
cfg.model.grad_checkpointing = True
cfg.model.mix_beta = 0.5
cfg.model.manifold_mixup = True

# others
cfg.seed = 42
cfg.device = 'cuda'
cfg.lr = 1.0e-3
cfg.wd = 1.0e-3
cfg.min_lr = 5.0e-5
cfg.warmup_lr = 1.0e-5
cfg.warmup_epochs = 3
cfg.warmup = 1
cfg.epochs = 50
cfg.eval_intervals = 5
cfg.mixed_precision = True
cfg.ema_start_epoch = 1


def rescale_layer_norm(model, state_dict):
    model_state_dict = model.state_dict()
    for k, v in state_dict.items():
        new_shape = model_state_dict[k].shape
        old_shape = v.shape
        if new_shape != old_shape:
            print(f'rescale {k} from {old_shape} -> {new_shape}')
            state_dict[k] = torch.nn.functional.interpolate(
                v[None, None], new_shape, mode='bilinear').squeeze()
    return state_dict


def get_model(cfg, weight_path=None):
    model = Net(cfg.model)
    if cfg.model.resume_exp is not None:
        weight_path = os.path.join(
            cfg.root, 'output', cfg.model.resume_exp, f'best_fold{cfg.fold}.pth')
    if weight_path is not None:
        state_dict = torch.load(weight_path, map_location='cpu')
        epoch = state_dict['epoch']
        model_key = 'model_ema'
        if model_key not in state_dict.keys():
            model_key = 'model'
            print(f'load epoch {epoch} model from {weight_path}')
        else:
            print(f'load epoch {epoch} ema model from {weight_path}')
        if cfg.model.rescale_layer_norm:
            state_dict[model_key] = rescale_layer_norm(
                model, state_dict[model_key])

        model.load_state_dict(state_dict[model_key])

    return model.to(cfg.device)


def save_val_results(targets, preds, save_path):
    num_classes = targets.shape[1]
    df = pd.DataFrame()
    for c in range(num_classes):
        df[f'target_{c}'] = targets[:, c]
        df[f'pred_{c}'] = preds[:, c]
    df.to_csv(save_path, index=False)


def post_process(val_keys, val_cls_preds, val_reg_preds, val_masks=None, score_threshold=0.01, nms_thresholds=(12, 6, 6)):
    FPS = 25.0
    event_classes = [
        'challenge',
        'play',
        'throwin'
    ]

    has_mask = val_masks is not None

    val_keys = pd.Series(val_keys)
    val_videos = val_keys.map(lambda x: "_".join(x.split('_')[:2]))
    unique_val_videos = sorted(val_videos.unique())

    records = []
    for video in unique_val_videos:
        video_index = (val_videos == video)
        video_cls_preds = val_cls_preds[video_index]
        video_reg_preds = val_reg_preds[video_index]

        video_cls_preds = np.transpose(
            video_cls_preds, [0, 2, 1]).reshape(-1, 3)
        video_reg_preds = np.transpose(
            video_reg_preds, [0, 2, 1]).reshape(-1, 3)

        if has_mask:
            video_masks = val_masks[video_index]
            video_masks = np.transpose(video_masks, [0, 2, 1]).reshape(-1, 3)

        for c, (class_name, nms_th) in enumerate(zip(event_classes, nms_thresholds)):
            this_video_cls_preds = video_cls_preds[:, c]
            this_video_reg_preds = video_reg_preds[:, c]

            if has_mask:
                this_video_masks = video_masks[:, c]
                predictions = np.where(
                    (this_video_cls_preds > score_threshold) & (this_video_masks == 1))[0]
            else:
                predictions = np.where(
                    (this_video_cls_preds > score_threshold))[0]
            offsets = this_video_reg_preds[predictions]
            offsets = offsets * FPS  # convert to frame scale
            scores = this_video_cls_preds[predictions]
            # predictions = predictions + offsets

            keep = nms(predictions, scores, nms_th)
            predictions = predictions[keep]
            scores = scores[keep]
            predictions = predictions / FPS

            for prediction, score in zip(predictions, scores):
                records.append((video, prediction, class_name, score))

    result_df = pd.DataFrame(data=records, columns=[
                             'video_id', 'time', 'event', 'score'])
    return result_df


def get_optimizer(model, cfg):
    def exclude(
        n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

    def include(n, p): return not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(
        n, p) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": cfg.wd},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    return optimizer


def train(cfg, fold):
    os.makedirs(str(cfg.output_dir + "/"), exist_ok=True)
    cfg.fold = fold
    mode = 'disabled' if cfg.debug else None
    wandb.init(project=cfg.project,
               name=f'{cfg.exp_name}_fold{fold}', config=cfg, reinit=True, mode=mode)
    set_seed(cfg.seed)
    train_dataloader = get_trainval_dataloader(cfg.train, fold)
    cfg.model.samples_per_class = train_dataloader.dataset.samples_per_class
    model = get_model(cfg)

    if cfg.model.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = ModelEmaV2(model, decay=0.999)

    optimizer = get_optimizer(model, cfg)
    steps_per_epoch = len(train_dataloader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cfg.epochs*steps_per_epoch,
        lr_min=cfg.min_lr,
        warmup_lr_init=cfg.warmup_lr,
        warmup_t=cfg.warmup_epochs*steps_per_epoch,
        k_decay=1.0,
    )

    scaler = GradScaler(enabled=cfg.mixed_precision)
    init_epoch = 0
    best_val_score = 0
    ckpt_path = f"{cfg.output_dir}/last_fold{fold}.pth"
    if cfg.resume and os.path.exists(ckpt_path):
        model, optimizer, init_epoch, best_val_score, scheduler, scaler, model_ema = resume_checkpoint(
            f"{cfg.output_dir}/last_fold{fold}.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            model_ema
        )

    cfg.curr_step = 0
    i = init_epoch * steps_per_epoch

    optimizer.zero_grad()
    for epoch in range(init_epoch, cfg.epochs):
        set_seed(cfg.seed + epoch)

        cfg.curr_epoch = epoch

        progress_bar = tqdm(range(len(train_dataloader)),
                            leave=False, dynamic_ncols=True)
        tr_it = iter(train_dataloader)

        cls_losses = []
        reg_losses = []
        targets = []
        cls_preds = []
        reg_preds = []
        masks = []

        gc.collect()

        # ==== TRAIN LOOP
        for itr in progress_bar:
            i += 1
            cfg.curr_step += cfg.train.batch_size

            model.train()
            torch.set_grad_enabled(True)

            inputs = next(tr_it)
            inputs = batch_to_device(inputs, cfg.device, cfg.mixed_precision)

            optimizer.zero_grad()
            with autocast(enabled=cfg.mixed_precision):
                outputs = model(inputs)
                loss_dict = model.get_loss(outputs, inputs)
                loss = loss_dict['loss']

            cls_losses.append(loss_dict['cls'].item())
            reg_losses.append(loss_dict['reg'].item())
            targets.append(inputs['labels'].cpu().numpy())
            cls_preds.append(outputs['cls'].sigmoid().detach().cpu().numpy())
            reg_preds.append(outputs['reg'].detach().cpu().numpy())
            masks.append(inputs['masks'].cpu().numpy())

            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if model_ema is not None:
                model_ema.update(model)

            if scheduler is not None:
                scheduler.step(i)

            avg_cls_loss = np.mean(cls_losses[-10:])
            avg_reg_loss = np.mean(reg_losses[-10:])
            lr = optimizer.param_groups[0]['lr']
            progress_bar.set_description(
                f"step:{i} cls_loss: {avg_cls_loss:.4f} reg_loss: {avg_reg_loss:.4f} lr:{lr:.6}")

        targets = np.concatenate(targets, axis=0).reshape(-1)
        cls_preds = np.concatenate(cls_preds, axis=0).reshape(-1)
        masks = np.concatenate(masks, axis=0).reshape(-1)
        score = average_precision_score(
            targets == 1, cls_preds, sample_weight=masks)

        lr = optimizer.param_groups[0]['lr']

        all_results = {
            'epoch': epoch,
            'lr': lr,
        }
        train_results = {
            'cls_loss': avg_cls_loss,
            'reg_loss': avg_reg_loss,
            'score': score,
        }
        val_results = {}
        log_results(all_results, train_results, val_results)

        if epoch % 5 == 0:
            checkpoint = create_checkpoint(
                model, optimizer, epoch, scheduler=scheduler, scaler=scaler, score=best_val_score,
                model_ema=model_ema
            )
            torch.save(checkpoint, f"{cfg.output_dir}/epoch{epoch:02}.pth")

        checkpoint = create_checkpoint(
            model, optimizer, epoch, scheduler=scheduler, scaler=scaler, model_ema=model_ema)
        torch.save(checkpoint, f"{cfg.output_dir}/last_fold{fold}.pth")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./", type=str)
    parser.add_argument("--device_id", "-d", default="0", type=str)
    parser.add_argument("--start_fold", "-s", default=0, type=int)
    parser.add_argument("--end_fold", "-e", default=5, type=int)
    parser.add_argument("--validate", "-v", action="store_true")
    parser.add_argument("--infer", "-i", action="store_true")
    parser.add_argument("--debug", "-db", action="store_true")
    parser.add_argument("--resume", "-r", action="store_true")
    return parser.parse_args()


def update_cfg(cfg, args, fold):
    if args.debug:
        cfg.debug = True
        set_debugger()

    cfg.fold = fold

    if args.resume:
        cfg.resume = True

    cfg.root = args.root

    cfg.output_dir = os.path.join(args.root, cfg.output_dir)

    if cfg.model.resume_exp is not None:
        cfg.model.pretrained_path = os.path.join(
            cfg.root, 'output', cfg.model.resume_exp, f'best_fold{cfg.fold}.pth')

    return cfg


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    fold = -1
    cfg = update_cfg(cfg, args, fold)
    train(cfg, fold)