
import torch
import torch.nn as nn
import types
from timm.models.helpers import checkpoint_seq

import timm
from models.losses import CenterNetLoss

from models.temporal_shift import TemporalShift

from .common import load_wraped_model_ckpt


class SpatialAttention3d(nn.Module):
    def __init__(self, n_channels, duration=8, kernel_size=3, pooling=True):
        super(SpatialAttention3d, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.n_channels = n_channels
        self.duration = duration
        self.pooling = pooling

        self.conv1 = nn.Conv2d(self.n_channels*2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        if pooling:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        _, c, h, w = x.shape
        x = x.view(-1, self.duration, c, h, w)

        # avg and max in time dim
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = att.reshape(-1, 2*c, h, w)
        att = self.conv1(att)
        att = self.sigmoid(att)[:, None, ...]
        x = x * att
        x = x.view(-1, c, h, w)
        if self.pooling:
            x = self.pool(x)
        return x


class BallNet(nn.Module):
    """grad checkpointing"""

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.n_classes = model_cfg.num_classes
        self.duration = model_cfg.duration
        self.cls_weight = model_cfg.cls_weight

        self.model = timm.create_model(
            model_cfg.model_name,
            pretrained=model_cfg.pretrained,
            num_classes=0,
            drop_rate=model_cfg.drop,
            drop_path_rate=model_cfg.drop_path,
        )

        def forward_features(self, x):
            x = self.conv_stem(x)
            x = self.bn1(x)
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(self.blocks, x, flatten=False)
            else:
                x = self.blocks(x)
            return x
        funcType = types.MethodType
        self.model.forward = funcType(forward_features, self.model)

        self._convert_temporal_shift()

        if model_cfg.pretrained_path is not None:
            print(f'load backbone from {model_cfg.pretrained_path}')
            load_wraped_model_ckpt(self.model, model_cfg.pretrained_path)

        self.n_channels = self.model.feature_info[-1]['num_chs']
        self.spatial_att = SpatialAttention3d(self.n_channels, self.duration, pooling=False)

        self.head = nn.Sequential(
            nn.Conv2d(self.n_channels, 128,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))

        self.cls_loss = CenterNetLoss()

    def _convert_temporal_shift(self, n_div=8):
        for i, block in enumerate(self.model.blocks):
            if self.model_cfg.temporal_shift_dwc:
                for j, dwc in enumerate(block):
                    block[j] = TemporalShift(dwc, n_segment=self.duration, n_div=n_div)
                self.model.blocks[i] = block
            else:
                self.model.blocks[i] = TemporalShift(block, n_segment=self.duration, n_div=n_div)

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable=True)

    def forward(self, inputs):
        images = inputs['features']
        b, t, c, h, w = images.shape
        x = self.model(images.view(b*t, c, h, w))
        x = self.spatial_att(x)
        logits = self.head(x)
        bt, _, out_h, out_w = logits.shape
        logits = logits.view(b, t, out_h, out_w)
        outputs = {
            'pred': logits,
        }
        return outputs

    def get_loss(self, outputs, inputs):
        cls_loss = self.cls_loss(outputs['pred'], inputs['labels'], inputs['masks'])
        loss_dict = {
            'loss': cls_loss,
            'cls': cls_loss,
        }
        return loss_dict
