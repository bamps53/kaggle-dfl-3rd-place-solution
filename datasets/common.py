import pandas as pd
import cv2
import numpy as np
import torch
from PIL import Image

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

def normalize_img(img):
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
    img = img.astype(np.float32)
    img -= mean
    img *= np.reciprocal(std, dtype=np.float32)
    return img

def denormalize_img(img):
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
    img = img.astype(np.float32)
    img *= std
    img += mean
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def to_torch_tensor(img):
    return torch.from_numpy(img.transpose((2, 0, 1)))

def to_numpy(img_tensor):
    return np.transpose(img_tensor.cpu().numpy(), [1, 2, 0])


def load_torch_image(image_path):
    img = cv2.imread(image_path)
    img = img[:, :, ::-1]
    img = normalize_img(img)
    img = to_torch_tensor(img)
    return img


def load_pil_image(image_path):
    img = Image.open(image_path)
    return img


def load_cv2_image(image_path):
    img = cv2.imread(image_path)
    assert img is not None, image_path
    img = img[:, :, ::-1]
    return img

def torch_pad_if_needed(x, max_len):
    if len(x) == max_len:
        return x
    b = len(x)
    res = x.shape[1:]

    num_pad = max_len - b
    pad = torch.zeros((num_pad, *res)).to(x)
    return torch.cat([x, pad], dim=0)