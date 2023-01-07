import argparse
import gc
import os
import importlib

import numpy as np
import torch
from timm.scheduler import CosineLRScheduler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    # import training only modules
    import wandb
except:
    print('wandb is not installed.')

from datasets.segm import get_train_dataloader, get_val_dataloader
from models.ball_net import BallNet as Net
from utils.common import (batch_to_device, create_checkpoint, log_results,
                          resume_checkpoint, set_seed)
from utils.debugger import set_debugger
from utils.ema import ModelEmaV2

from models.common import rescale_layer_norm
from optimizers.common import get_optimizer


def get_model(cfg, weight_path=None):
    model = Net(cfg.model)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", type=str)
    parser.add_argument("--root", default="./", type=str)
    parser.add_argument("--device_id", "-d", default="0", type=str)
    parser.add_argument("--start_fold", "-s", default=0, type=int)
    parser.add_argument("--end_fold", "-e", default=1, type=int)
    parser.add_argument("--validate", "-v", action="store_true")
    parser.add_argument("--infer", "-i", action="store_true")
    parser.add_argument("--debug", "-db", action="store_true")
    parser.add_argument("--resume", "-r", action="store_true")
    return parser.parse_args()


def setup_cfg(args):
    cfg = importlib.import_module(args.config_path).cfg

    if args.debug:
        cfg.debug = True
        set_debugger()
    if args.resume:
        cfg.resume = True
    cfg.root = args.root
    cfg.output_dir = os.path.join(args.root, cfg.output_dir)

    return cfg


if __name__ == "__main__":
    args = parse_args()

    cfg = setup_cfg(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    for fold in range(args.start_fold, args.end_fold):
        train(cfg, fold)
