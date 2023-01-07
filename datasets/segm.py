import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2

from .common import setup_df, normalize_img, to_torch_tensor, load_cv2_image, load_cv2_image
from .video_point_transform import VideoPointTransforms
FRAMES_PER_VIDEO = 750  # roughly 750


def get_label_dict(df):
    df['video_frame'] = df['video_id'] + "_" + df['frame'].map(lambda x: f"{x:06}")
    # label_dict = {k: v for k, v in zip(df['video_frame'], df[['x', 'y', 'w', 'h']].values)}
    label_dict = {k: v for k, v in zip(df['video_frame'], df[['cx', 'cy']].values)}
    return label_dict

# from CenterNet


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

# from CenterNet


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_centernet_label(img_h, img_w, center, down_ratio, radius=2, k=1):
    label_h, label_w = img_h // down_ratio, img_w // down_ratio
    heatmap = np.zeros((label_h, label_w))
    if center is None:
        return heatmap
    cx, cy = center
    cx /= down_ratio
    cy /= down_ratio
    heatmap = draw_umich_gaussian(heatmap, (cx, cy), radius, k)
    return heatmap


def get_frames(center_frame, before_duration, after_duration):
    frames = [center_frame + i for i in range(-before_duration, after_duration + 1)]
    return frames


def draw_labels(label_list, image_list, down_ratio):
    # img_w, img_h = image_list[0].size
    img_h, img_w = image_list[0].shape[:2]
    heatmap_list = [draw_centernet_label(img_h, img_w, center, down_ratio, radius=2, k=1) for center in label_list]
    return heatmap_list


class TrainDataset(Dataset):
    def __init__(self, cfg, fold, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = setup_df(cfg.df_path, fold, mode)
        self.label_dict = get_label_dict(self.df)
        self.original_image_size = self.cfg.original_image_size

        self.down_ratio = self.cfg.down_ratio

        self.duration = self.cfg.duration
        # assert self.duration % 2 == 1, f'duration has to be odd number.'
        self.before_duration = (self.duration - 1) // 2
        self.after_duration = self.duration - 1 - self.before_duration

        self.center_indices = self.df.index[
            (self.df['frame'] > self.duration) &
            (self.df['frame'] < (FRAMES_PER_VIDEO - self.duration))
        ]

        self.transforms = VideoPointTransforms()

    def get_frame_path(self, video_id, frame):
        assert (frame >= 0) and (frame <= FRAMES_PER_VIDEO)
        frame_path = f'{self.cfg.data_dir}/{video_id}/{self.cfg.folder_name}/{frame:06}.jpg'
        return frame_path

    def load_sequence(self, video_id, frames):
        image_paths = [self.get_frame_path(video_id, frame) for frame in frames]
        image_list = [load_cv2_image(p) for p in image_paths]
        return image_list

    def get_labels(self, video_id, frames):
        labels = []
        for frame in frames:
            key = f'{video_id}_{frame:06}'
            labels.append(self.label_dict.get(key, None))
        masks = [label is not None for label in labels]
        return labels, masks

    def augment(self, image_list, heatmap_list):
        if self.mode != 'train':
            return image_list, heatmap_list

        num_images = len(image_list)
        concat_list = image_list + heatmap_list
        if np.random.uniform() < 0.5:
            concat_list = [cv2.flip(img, 1) for img in concat_list]
        image_list = concat_list[:num_images]
        heatmap_list = concat_list[num_images:]
        return image_list, heatmap_list

    def resize_image_and_label(self, image_list, label_list):
        # before_shape = image_list[0].shape
        before_shape = self.original_image_size
        height, width = self.cfg.image_size
        image_list = [cv2.resize(img, (width, height)) for img in image_list]
        after_shape = image_list[0].shape
        h_scale = after_shape[0] / before_shape[0]
        w_scale = after_shape[1] / before_shape[1]

        def _resize_label(cxcy):
            if cxcy is None:
                return cxcy
            cx, cy = cxcy
            cx *= w_scale
            cy *= h_scale
            return np.array([cx, cy])
        label_list = [_resize_label(cxcy) for cxcy in label_list]
        return image_list, label_list

    def __len__(self):
        return len(self.center_indices)

    def __getitem__(self, idx):
        center_frame_row = self.df.iloc[self.center_indices[idx]]
        frames = get_frames(center_frame_row.frame, self.before_duration, self.after_duration)
        image_list = self.load_sequence(center_frame_row.video_id, frames)
        label_list, mask_list = self.get_labels(center_frame_row.video_id, frames)
        if self.mode == 'train':
            image_list, label_list = self.transforms(image_list, label_list)
        image_list, label_list = self.resize_image_and_label(image_list, label_list)
        heatmap_list = draw_labels(label_list, image_list, self.down_ratio)
        if self.mode == 'train':
            image_list, heatmap_list = self.augment(image_list, heatmap_list)

        images = [to_torch_tensor(normalize_img(
            np.asarray(img))) for img in image_list]
        images = torch.stack(images, dim=0)

        heatmaps = torch.from_numpy(np.stack(heatmap_list))
        masks = np.array(mask_list)

        inputs = {
            'features': images,
            'labels': heatmaps,
            'masks': masks
        }
        return inputs


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


def get_val_dataloader(cfg, fold):
    dataset = TrainDataset(cfg, fold=fold, mode="valid")
    val_dataloader = DataLoader(
        dataset,
        sampler=None,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False
    )
    return val_dataloader
