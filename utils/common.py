import gc
import pandas as pd
import numpy as np
import random
import os
import torch
import wandb

def setup_df(df_path, fold, mode):
    df = pd.read_csv(df_path)
    if mode == "train":
        index = df.folds != fold
    elif mode == 'valid':  # 'valid
        index = df.folds == fold
    else:
        index = df.index
    df = df.loc[index]
    df = df.reset_index(drop=True)
    return df


def pad_if_needed(x, max_len):
    if len(x) == max_len:
        return x
    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0, 0) for _ in range(n_dim - 1)]
    return np.pad(x, pad_width=pad_widths)


def torch_pad_if_needed(x, max_len):
    if len(x) == max_len:
        return x
    b = len(x)
    res = x.shape[1:]

    num_pad = max_len - b
    pad = torch.zeros((num_pad, *res)).to(x)
    return torch.cat([x, pad], dim=0)


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None, score=None, model_ema=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    if score is not None:
        checkpoint['score'] = score
    
    if model_ema is not None:
        checkpoint['model_ema'] = model_ema.module.state_dict()

    return checkpoint

def resume_checkpoint(ckpt_path, model, optimizer, scheduler=None, scaler=None, model_ema=None):
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt.get('epoch', 0) + 1
    score = ckpt.get('score', 0)
    
    print(f'resume training from {ckpt_path}')
    print(f'start training from epoch={epoch}')
    
    if scheduler is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    if scaler is not None:
        scaler.load_state_dict(ckpt['scaler'])
    if model_ema is not None:
        model_ema.module.load_state_dict(ckpt['model_ema'])
    ret = (model, optimizer, epoch, score, scheduler, scaler, model_ema)
    
    del ckpt
    gc.collect()
    torch.cuda.empty_cache() 
    return ret
    
    
def batch_to_device(batch, device, mixed_precision=False):
    batch_dict = {}
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch_dict[k] = batch[k].to(device, non_blocking=True)
        else:
            batch_dict[k] = batch[k]
            
        if mixed_precision and (k == 'features'):
            batch_dict[k] = batch_dict[k].half()
        elif isinstance(batch[k], torch.Tensor):
            batch_dict[k] = batch_dict[k].float()
    return batch_dict

def nms(predictions, scores, nms_threshold):
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        keep_pred = predictions[[i]]
        other_preds = predictions[order]
        enough_far = np.abs(other_preds - keep_pred) > nms_threshold
        order = order[enough_far]
    return keep


def log_results(all_results, train_results, val_results):
    def _add_text(text, key, value):
        if isinstance(value, float):
            text += f'{key}:{value:.3} '
        elif isinstance(value, (int, str)):
            text += f'{key}:{value} '
        else:
            print(key, value)
            # raise NotImplementedError
        return text

    text = "train "
    for k, v in all_results.items():
        text = _add_text(text, k, v)
    for k, v in train_results.items():
        text = _add_text(text, k, v)
    print(text)

    text = "valid "
    for k, v in val_results.items():
        text = _add_text(text, k, v)
    print(text)

    all_results.update({f'train_{k}': v for k, v in train_results.items()})
    all_results.update({f'val_{k}': v for k, v in val_results.items()})
    wandb.log(all_results)
