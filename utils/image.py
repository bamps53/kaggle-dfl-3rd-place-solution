import numpy as np
import cv2
import torch
from PIL import Image

def normalize_img(img):
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
    img = img.astype(np.float32)
    img -= mean
    img *= np.reciprocal(std, dtype=np.float32)
    return img


def to_torch_tensor(img):
    return torch.from_numpy(img.transpose((2, 0, 1)))


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
    img = img[:, :, ::-1]
    return img