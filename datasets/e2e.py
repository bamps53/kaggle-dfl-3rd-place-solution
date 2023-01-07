import json
import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import video_transforms

from .common import setup_df, normalize_img, to_torch_tensor, load_pil_image, torch_pad_if_needed

EVENT_CLASSES = [
    'challenge',
    'play',
    'throwin'
]
FPS = 25.0


def gaussian_kernel(length, sigma=3):
    x = np.ogrid[-length:length+1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def pad_if_needed(x, max_len):
    if len(x) == max_len:
        return x
    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0, 0) for _ in range(n_dim - 1)]
    return np.pad(x, pad_width=pad_widths)


def load_image_paths(video_names, dir_name):
    features_dict = {}
    for video_name in video_names:
        video_dir = f"{dir_name}/{video_name}"
        features_dict[video_name] = sorted(glob.glob(f'{video_dir}/*.jpg'))
    return features_dict


def create_offset_labels(labels, length=10):
    offset_labels = np.zeros_like(labels)
    for pos in np.where(labels == 1)[0]:
        for off in np.ogrid[-length:length+1]:
            if 0 <= pos+off < len(offset_labels):
                offset_labels[pos+off] = -off
    offset_labels /= FPS  # convert to time scale
    return offset_labels


def load_labels(video_names, dir_name, offset, sigma):
    labels_dict = {}
    offset_labels_dict = {}
    label_pos = []
    for video_name in video_names:
        label_path = f"{dir_name}/{video_name}.npy"
        labels = np.load(label_path)

        for pos in np.where(labels.sum(axis=1))[0]:
            label_pos.append((video_name, pos))

        offset_labels = np.zeros_like(labels)
        for c in range(labels.shape[1]):
            offset_labels[:, c] = create_offset_labels(
                labels[:, c], length=offset)
            labels[:, c] = np.convolve(
                labels[:, c], gaussian_kernel(offset, sigma), mode='same')  # 1.5 sec

        labels = np.clip(labels, 0, 1)
        labels_dict[video_name] = labels
        offset_labels_dict[video_name] = offset_labels

    return labels_dict, offset_labels_dict, label_pos


def load_masks(video_names, dir_name):
    masks_dict = {}
    for video_name in video_names:
        mask_path = f"{dir_name}/{video_name}_mask.npy"
        masks_dict[video_name] = np.load(mask_path)
    return masks_dict


def get_random_crop_position(pos, duration):
    start = random.randint(-duration, 0)
    end = start + duration
    start = pos + start
    end = pos + end
    return start, end

def load_image_from_paths(video_dir, start, end):
    image_paths = [f'{video_dir}/{frame:06}.jpg' for frame in range(start, end)]
    return [load_pil_image(f) for f in image_paths if os.path.exists(f)]


def load_chunk_labels(video_names, dir_name, duration, offset):
    labels_dict = {}
    offset_labels_dict = {}
    for video_name in video_names:
        video_path = f"{dir_name}/{video_name}.npy"
        labels = np.load(video_path)
        offset_labels = np.zeros_like(labels)
        for c in range(labels.shape[1]):
            offset_labels[:, c] = create_offset_labels(
                labels[:, c], length=offset)

        num_chunks = (len(labels) // duration) + 1
        for i in range(num_chunks):
            chunk_labels = labels[i*duration:(i+1)*duration]
            chunk_labels = pad_if_needed(chunk_labels, duration)
            labels_dict[f"{video_name}_{i:04}"] = chunk_labels

            chunk_offset_labels = offset_labels[i*duration:(i+1)*duration]
            chunk_offset_labels = pad_if_needed(chunk_offset_labels, duration)
            offset_labels_dict[f"{video_name}_{i:04}"] = chunk_offset_labels
    return labels_dict, offset_labels_dict


def load_chunk_masks(video_names, dir_name, duration):
    masks_dict = {}
    for video_name in video_names:
        video_path = f"{dir_name}/{video_name}_mask.npy"
        masks = np.load(video_path)
        num_chunks = (len(masks) // duration) + 1
        for i in range(num_chunks):
            chunk_masks = masks[i*duration:(i+1)*duration]
            chunk_masks = pad_if_needed(chunk_masks, duration)
            masks_dict[f"{video_name}_{i:04}"] = chunk_masks
    return masks_dict


def load_json(json_path):
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    else:
        ann = {}
    return ann


class TrainDataset(Dataset):
    def __init__(self, cfg, fold, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = setup_df(cfg.df_path, fold, mode)
        self.all_df = setup_df(cfg.all_df_path, fold, mode)
        self.all_df = self.all_df[['video_id',
                                   'time', 'event']].reset_index(drop=True)
        self.samples_per_class = self.all_df['event'].value_counts()[EVENT_CLASSES].values
        self.video_names = self.df['video_id'].unique().tolist()
        self.video_dir = self.cfg.video_feature_dir
        self.label_dir = self.cfg.label_dir
        self.duration = self.cfg.duration
        self.offset = self.cfg.offset
        self.sigma = self.cfg.sigma

        height, width = cfg.image_size

        self.labels, self.offset_labels, self.label_pos = load_labels(
            self.video_names, self.label_dir, self.offset, self.sigma)
        self.masks = load_masks(self.video_names, self.label_dir)
        if cfg.transforms == None:
            self.transforms = video_transforms.Compose([
                video_transforms.RandomHorizontalFlip(),
                video_transforms.RandomRotation(5),
                video_transforms.ColorJitter(brightness=0.2, contrast=0.1),
                video_transforms.RandomCrop((int(height*0.9), int(width*0.9))),
                video_transforms.Resize((height, width)),
            ])
        else:
            self.transforms = cfg.transforms

    def __len__(self):
        return len(self.label_pos)

    def __getitem__(self, idx):
        video_name, pos = self.label_pos[idx]
        video_dir = f"{self.video_dir}/{video_name}"
        # image_paths = self.image_paths[video_name]
        labels = self.labels[video_name]
        off_labels = self.offset_labels[video_name]
        masks = self.masks[video_name]
        count = 0
        if np.random.uniform() < self.cfg.bg_sampling_rate:
            pos = random.sample(np.where(masks == 1)[0].tolist(), 1)[0]
        while True:
            start, end = get_random_crop_position(pos, self.duration)
            image_list = load_image_from_paths(video_dir, start, end)
            image_list = self.transforms(image_list)
            image_list = [to_torch_tensor(normalize_img(
                np.asarray(img))) for img in image_list]
            images = torch.stack(image_list, dim=0)
            if len(images) < self.duration:
                count += 1
                if count > 100:
                    raise ValueError()
                print('retry:', count)
                continue
            else:
                this_masks = np.tile(masks[start:end][None, ], [3, 1])
                this_labels = np.transpose(labels[start:end], [1, 0])
                this_off_labels = np.transpose(off_labels[start:end], [1, 0])
                this_off_masks = np.logical_or(
                    (this_off_labels != 0), (this_labels == 1))
                assert len(images) == this_labels.shape[1], (len(images), this_labels.shape[1])
                inputs = {
                    'features': images,
                    'labels': this_labels,
                    'off_labels': this_off_labels,
                    'off_masks': this_off_masks,
                    'masks': this_masks,
                }
                return inputs


class ValidDataset(Dataset):
    def __init__(self, cfg, fold, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = setup_df(cfg.df_path, fold, mode)
        self.all_df = setup_df(cfg.all_df_path, fold, mode)
        self.all_df = self.all_df[['video_id',
                                   'time', 'event']].reset_index(drop=True)
        self.video_names = self.df['video_id'].unique().tolist()
        self.video_dir = self.cfg.video_feature_dir
        self.label_dir = self.cfg.label_dir
        self.duration = self.cfg.duration
        self.offset = self.cfg.offset

        height, width = cfg.image_size

        self.labels, self.off_labels = load_chunk_labels(
            self.video_names, self.label_dir, self.duration, self.offset)
        self.masks = load_chunk_masks(
            self.video_names, self.label_dir, self.duration)
        self.keys = list(self.labels.keys())

        if cfg.transforms == None:
            self.transforms = video_transforms.Compose([
                video_transforms.Resize((height, width)),
            ])
        else:
            self.transforms = cfg.transforms

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        chunk_labels = self.labels[key]
        chunk_off_labels = self.off_labels[key]
        chunk_masks = self.masks[key]

        video = '_'.join(key.split('_')[:2])
        position = int(key.split('_')[-1])
        start = position * self.duration
        end = (position + 1) * self.duration
        video_dir = f"{self.video_dir}/{video}"
        image_list = load_image_from_paths(video_dir, start, end)
        image_list = self.transforms(image_list)
        image_list = [to_torch_tensor(normalize_img(
            np.asarray(img))) for img in image_list]
        chunk_images = torch.stack(image_list, dim=0)
        chunk_images = torch_pad_if_needed(chunk_images, self.duration)

        chunk_masks = np.tile(chunk_masks[None, ], [3, 1])  # (3, 512)
        chunk_labels = np.transpose(chunk_labels, [1, 0])  # (3, 512)
        chunk_off_labels = np.transpose(chunk_off_labels, [1, 0])  # (3, 512)
        chunk_off_masks = np.logical_or(
            (chunk_off_labels != 0), (chunk_labels == 1))

        assert len(chunk_images) == chunk_labels.shape[1], (len(chunk_images), chunk_labels.shape[1])
        inputs = {
            'features': chunk_images,
            'labels': chunk_labels,
            'masks': chunk_masks,
            'off_labels': chunk_off_labels,
            'off_masks': chunk_off_masks,
            'keys': key,
        }
        return inputs


class VideoDataset(Dataset):
    def __init__(self, cfg, video_path):

        self.cfg = cfg
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        self.num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def __len__(self):
        return int(self.num_frames)

    def __getitem__(self, idx):
        _, img = self.cap.read()
        img = img[:, :, ::-1]
        img = cv2.resize(img, (self.cfg.image_size[1], self.cfg.image_size[0]))
        img = normalize_img(img)
        img = to_torch_tensor(img)
        return img


def get_trainval_dataloader(cfg, fold):
    dataset = TrainDataset(cfg, fold=fold, mode="trainval")
    train_dataloader = DataLoader(
        dataset,
        sampler=None,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True
    )
    return train_dataloader


def get_train_dataloader(cfg, fold):
    dataset = TrainDataset(cfg, fold=fold, mode="train")
    train_dataloader = DataLoader(
        dataset,
        sampler=None,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True
    )
    return train_dataloader


def get_full_val_dataloader(cfg, fold):
    dataset = ValidDataset(cfg, fold=fold, mode="valid")
    val_dataloader = DataLoader(
        dataset,
        sampler=None,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False
    )
    return val_dataloader


def get_video_dataloader(cfg, video_path, image_sizes=None, batch_size=None, num_workers=0):
    batch_size = batch_size or cfg.test.batch_size
    dataset = VideoDataset(cfg, video_path)
    test_dataloader = DataLoader(
        dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return test_dataloader