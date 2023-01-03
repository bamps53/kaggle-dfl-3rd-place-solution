import argparse
import gc
import os
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import torch
from datasets.common import denormalize_img, to_numpy, torch_pad_if_needed
from datasets.segm import get_image_dataloader
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    # import training only modules
    import wandb
except:
    print('wandb is not installed.')

try:
    from torchvideotransforms import video_transforms
except:
    video_transforms = None
    print('torchvideotransforms is not installed.')


from configs.base_segm import cfg
from datasets.common import torch_pad_if_needed
from datasets.segm import get_train_dataloader, get_val_dataloader
from models.unet import GCBallNet as Net
from utils.common import (batch_to_device, create_checkpoint, log_results,
                          resume_checkpoint, set_seed)
from utils.debugger import set_debugger
from utils.ema import ModelEmaV2

FPS = 25.0
HEIGHT, WIDTH = 360, 640

cfg = deepcopy(cfg)
cfg.project = 'kaggle-dfl-pt'
cfg.exp_name = 'stage3_ball024_b3_aug_clip_no_warmup_0918_sn_360'
cfg.output_dir = f'output/{cfg.exp_name}'
cfg.debug = False

cfg.train.df_path = '../input/SoccerNet/tracking/folds_resized360.csv'
cfg.train.data_dir = '../input/SoccerNet/tracking'
cfg.train.folder_name = 'img_resized360'
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
cfg.train.original_image_size = (360, 640)

cfg.valid.df_path = '../input/SoccerNet/tracking/folds_resized360.csv'
cfg.valid.data_dir = '../input/SoccerNet/tracking'
cfg.valid.folder_name = 'img_resized360'
cfg.valid.duration = 32
cfg.valid.offset = 5
cfg.valid.batch_size = 4
cfg.valid.num_workers = 4 if not cfg.debug else 0
cfg.valid.image_size = (HEIGHT, WIDTH)
cfg.valid.down_ratio = 32
cfg.valid.original_image_size = (360, 640)

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

cfg.model.model_name = 'tf_efficientnet_b3_ns'
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
cfg.model.temporal_shift_dwc = True

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
cfg.steps_per_epoch = 5000


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


def calc_score(y_pred, y_true):
    b, t, h, w = y_pred.shape
    y_pred = y_pred.view(b*t, h*w).argmax(1)
    y_true = y_true.view(b*t, h*w).argmax(1)

    y_pred = torch.stack([y_pred % w, torch.div(
        y_pred, w, rounding_mode='trunc')], dim=1)
    y_true = torch.stack([y_true % w, torch.div(
        y_true, w, rounding_mode='trunc')], dim=1)
    correct = (y_pred == y_true).all(1)
    dist = ((y_pred - y_true)**2).sum(dim=1)**0.5
    return correct, dist


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
    train_dataloader = get_train_dataloader(cfg.train, fold)
    model = get_model(cfg)
    if cfg.model.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None

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
    if cfg.resume:
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
        if (epoch >= cfg.ema_start_epoch) and (model_ema is not None):
            model_ema = ModelEmaV2(model, decay=0.999)

        set_seed(cfg.seed + epoch)

        cfg.curr_epoch = epoch

        this_steps_per_epoch = cfg.steps_per_epoch or steps_per_epoch
        progress_bar = tqdm(range(this_steps_per_epoch), dynamic_ncols=True)
        tr_it = iter(train_dataloader)

        cls_losses = []
        accuracies = []
        dists = []

        gc.collect()

        # ==== TRAIN LOOP
        for itr in progress_bar:
            i += 1
            cfg.curr_step += cfg.train.batch_size

            model.train()
            torch.set_grad_enabled(True)

            inputs = next(tr_it)
            inputs = batch_to_device(inputs, cfg.device)

            with autocast(enabled=cfg.mixed_precision):
                outputs = model(inputs)
                loss_dict = model.get_loss(outputs, inputs)
                loss = loss_dict['loss']

            cls_losses.append(loss_dict['cls'].item())

            optimizer.zero_grad()
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                print('loss nan detected.')

            if model_ema is not None:
                model_ema.update(model)

            if scheduler is not None:
                scheduler.step(i)

            avg_cls_loss = np.mean(cls_losses[-10:])
            lr = optimizer.param_groups[0]['lr']

            acc, dist = calc_score(outputs['pred'].detach(), inputs['labels'])
            accuracies.append(acc.cpu().numpy())
            dists.append(dist.cpu().numpy())
            avg_acc = np.mean(np.concatenate(accuracies[-10:]))
            avg_dist = np.mean(np.concatenate(dists[-10:]))
            progress_bar.set_description(
                f"cls_loss: {avg_cls_loss:.4f} acc: {avg_acc:.4f} dist: {avg_dist:.4f} lr:{lr:.6}")

        checkpoint = create_checkpoint(
            model, optimizer, epoch, scheduler=scheduler, scaler=scaler, model_ema=model_ema)
        torch.save(checkpoint, f"{cfg.output_dir}/last_fold{fold}.pth")

        if epoch % cfg.eval_intervals == 0:
            if model_ema is not None:
                val_results = run_full_eval(cfg, fold, model_ema.module)
            else:
                val_results = run_full_eval(cfg, fold, model)
        else:
            val_results = {}
        lr = optimizer.param_groups[0]['lr']

        all_results = {
            'epoch': epoch,
            'lr': lr,
        }
        train_results = {
            'cls_loss': avg_cls_loss,
            # 'reg_loss': avg_reg_loss,
            # 'score': score,
        }
        log_results(all_results, train_results, val_results)

        val_score = val_results.get('score', 0.0)
        if best_val_score < val_score:
            best_val_score = val_score
            checkpoint = create_checkpoint(
                model, optimizer, epoch, scheduler=scheduler, scaler=scaler, score=best_val_score,
                model_ema=model_ema
            )
            torch.save(checkpoint, f"{cfg.output_dir}/best_fold{fold}.pth")


def run_full_eval(cfg, fold, model=None, test_dataloader=None):

    if model is None:
        model = get_model(cfg)
        weight_path = f"{cfg.output_dir}/best_fold{fold}.pth"
        model.load_state_dict(torch.load(weight_path)['model'])
        print('load model from', weight_path)
    model.eval()
    torch.set_grad_enabled(False)

    if os.path.exists('../input/train_frames'):
        visualize_prediction(model, cfg.curr_epoch)

    if test_dataloader is None:
        test_dataloader = get_val_dataloader(cfg.valid, fold)

    cls_losses = []
    accuracies = []
    dists = []

    progress_bar = tqdm(range(len(test_dataloader)), dynamic_ncols=True)
    test_iter = iter(test_dataloader)
    for itr in progress_bar:
        inputs = next(test_iter)
        inputs = batch_to_device(inputs, cfg.device)
        with autocast(cfg.mixed_precision):
            outputs = model(inputs)
            loss_dict = model.get_loss(outputs, inputs)

        acc, dist = calc_score(outputs['pred'], inputs['labels'])

        cls_losses.append(loss_dict['cls'].item())
        accuracies.append(acc.cpu().numpy())
        dists.append(dist.cpu().numpy())

        avg_cls_loss = np.mean(cls_losses[-10:])
        avg_acc = np.mean(np.concatenate(accuracies[-10:]))
        avg_dist = np.mean(np.concatenate(dists[-10:]))
        progress_bar.set_description(
            f"cls_loss: {avg_cls_loss:.4f} acc: {avg_acc:.4f} dist: {avg_dist:.4f}")

    val_score = np.mean(np.concatenate(accuracies))
    val_dist = np.mean(np.concatenate(dists))
    val_loss = np.mean(cls_losses)
    results = {'score': val_score, 'loss': val_loss, 'dist': val_dist}
    return results


def alpha_blend(img, hm):
    h, w = img.shape[:2]
    hm = cv2.resize(hm, (w, h))
    hm = np.stack([hm, hm, hm], axis=-1)
    hm = (hm * 255).astype(np.uint8)
    dst = cv2.addWeighted(img, 0.4, hm, 0.6, 0)
    # plt.imshow(dst)
    return dst


def draw_area(img, cx, cy):
    x1 = int(cx * cfg.train.down_ratio)
    y1 = int(cy * cfg.train.down_ratio)
    x2 = int((cx+1) * cfg.train.down_ratio)
    y2 = int((cy+1) * cfg.train.down_ratio)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return img


def calc_ball_pos(y_pred):
    b, t, h, w = y_pred.shape
    y_pred = y_pred.view(b*t, h*w).argmax(1)
    y_pred = torch.stack([y_pred % w, torch.div(
        y_pred, w, rounding_mode='trunc')], dim=1)
    return y_pred


def visualize_prediction(model, epoch):
    fps = 25.0
    img_h, img_w = cfg.test.image_size
    out_h = int(cfg.test.image_size[0] / cfg.train.down_ratio)
    out_w = int(cfg.test.image_size[1] / cfg.train.down_ratio)

    data_dir = '../input/train_frames'
    df_path = '../input/folds_all.csv'
    df = pd.read_csv(df_path)
    df = df.query('event in ["start", "end"]').reset_index(drop=True)

    count = 0
    centers = defaultdict(list)
    for video_id, video_df in df.groupby('video_id'):
        # video loop
        image_dir = f'{data_dir}/{video_id}'
        start_end_pairs = video_df['frame'].values.reshape(-1, 2)
        for start, end in start_end_pairs:
            # interval loop
            frames = range(start, end + 1)
            test_dataloader = get_image_dataloader(cfg.test, image_dir, frames)
            save_path = f'{cfg.output_dir}/epoch{epoch}_{video_id}_{start}_{end}_ball.mp4'
            writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(
                    *"mp4v"), fps, (int(img_w), int(img_h))
            )
            for i, images in enumerate(test_dataloader):
                # batch loop
                images = images.to(cfg.device)
                original_len = len(images)
                images = torch_pad_if_needed(images, cfg.test.batch_size)
                bt, c, h, w = images.shape
                images = images.view(-1, cfg.test.duration, c, h, w)
                outputs = model({'features': images})
                heatmaps = outputs['pred'].sigmoid()
                cxcys = calc_ball_pos(heatmaps).cpu().numpy()
                cxcys = cxcys[:original_len]
                images = images.view(-1, c, h, w)[:original_len]
                heatmaps = heatmaps.view(-1, out_h, out_w)[:original_len]
                centers[f'{video_id}_{start}_{end}'].append(cxcys)

                for img, hm, (cx, cy) in zip(images, heatmaps, cxcys):
                    # frame loop
                    img = denormalize_img(to_numpy(img))
                    hm = hm.cpu().numpy()
                    img = alpha_blend(img, hm)
                    img = draw_area(img, cx, cy)
                    # plt.imshow(img)
                    # plt.show()
                    writer.write(img)
            writer.release()
            print(f'video saved to {save_path}')
            count += 1
            if count == 5:
                return
    return


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


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        cfg.debug = True
        set_debugger()
    if args.resume:
        cfg.resume = True
    cfg.root = args.root
    cfg.output_dir = os.path.join(args.root, cfg.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    for fold in range(args.start_fold, args.end_fold):
        if args.validate:
            run_full_eval(cfg, fold)
        elif args.infer:
            inference(cfg)
        else:
            train(cfg, fold)
