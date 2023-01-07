import random
from functools import partial

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import load_wraped_model_ckpt
from .losses import get_cls_loss, get_reg_loss
from .mixup import SeparateMultiMixup
from .temporal_shift import TemporalShift


class SEModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    
    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm=nn.BatchNorm1d, se=False, res=False):
        super().__init__()
        self.res = res
        if not mid_channels:
            mid_channels = out_channels
        if se:
            non_linearity = SEModule(out_channels)
        else:
            non_linearity = nn.ReLU(inplace=True)
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(out_channels),
            non_linearity
        )

    def forward(self, x):
        if self.res:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor, norm=nn.BatchNorm1d, se=False, res=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(scale_factor),
            DoubleConv(in_channels, out_channels, norm=norm, se=se, res=res)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2, norm=nn.BatchNorm1d):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=scale_factor, stride=scale_factor)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)
    
    
class ImageUNetLNMixup(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.n_channels = model_cfg.in_channels
        self.n_classes = model_cfg.num_classes
        bilinear = False
        self.duration = model_cfg.duration
        scale_factor = model_cfg.scale_factor
        self.drop_rate = model_cfg.drop_rate
        cls_loss_type = model_cfg.cls_loss_type
        self.cls_weight = model_cfg.cls_weight
        self.reg_weight = model_cfg.reg_weight
        self.model_name = model_cfg.model_name
        self.model_cfg = model_cfg
        
        self.model = timm.create_model(
            model_cfg.model_name,
            pretrained=model_cfg.pretrained,
            num_classes=0,
            drop_rate=model_cfg.drop,
            drop_path_rate=model_cfg.drop_path,
            # drop_block_rate=model_cfg.drop_block,
            )
            
        if model_cfg.grad_checkpointing:
            import types

            from timm.models.helpers import checkpoint_seq
            def forward_features(self, x):
                x = self.conv_stem(x)
                x = self.bn1(x)
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint_seq(self.blocks, x, flatten=False)
                else:
                    x = self.blocks(x)
                x = self.conv_head(x)
                x = self.bn2(x)
                return x
            funcType = types.MethodType
            self.model.forward_features = funcType(forward_features, self.model)
        
        if model_cfg.non_local:
            self._convert_non_local()
        if model_cfg.temporal_shift:
            self._convert_temporal_shift(model_cfg.n_div)
        if model_cfg.pretrained_path is not None:
            print(f'load backbone from {model_cfg.pretrained_path}')
            load_wraped_model_ckpt(self.model, model_cfg.pretrained_path)
            # ckpt = torch.load(model_cfg.pretrained_path, map_location='cpu')['model']
            # self.model.load_state_dict(ckpt, strict=False)

        if model_cfg.freeze_backbone:
            self.freeze_weights(freeze=['model'])
        
        if model_cfg.norm_type != 'ln':
            raise NotImplementedError()

        def create_layer_norm(channel, length):
            return nn.LayerNorm([channel, length])

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(self.n_channels, 64, norm=partial(create_layer_norm, length=self.duration))
        self.down1 = Down(64, 128, scale_factor, norm=partial(create_layer_norm, length=self.duration//2))
        self.down2 = Down(128, 256, scale_factor, norm=partial(create_layer_norm, length=self.duration//4))
        self.down3 = Down(256, 512, scale_factor, norm=partial(create_layer_norm, length=self.duration//8))
        self.down4 = Down(512, 1024 // factor, scale_factor, norm=partial(create_layer_norm, length=self.duration//16))
        self.up1 = Up(1024, 512 // factor, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration//8))
        self.up2 = Up(512, 256 // factor, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration//4))
        self.up3 = Up(256, 128 // factor, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration//2))
        self.up4 = Up(128, 64, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration))        
    
        self.cls = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, self.n_classes, kernel_size=1, padding=0),
            )
        self.reg = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, self.n_classes, kernel_size=1, padding=0),
            )

        if self.drop_rate > 0:
            print(f'turn on dropout with rate={self.drop_rate}')
            self.dropout = nn.Dropout(self.drop_rate)

        self.cls_loss = get_cls_loss(model_cfg)
        self.reg_loss = get_reg_loss(model_cfg)
    
        self.do_mixup = False
        self.do_manifold_mixup = False
        self.both_mixup = False
        if model_cfg.mix_beta > 0:
            if model_cfg.manifold_mixup:
                self.do_manifold_mixup = True
            else:
                self.do_mixup = True
            self.both_mixup = model_cfg.both_mixup
            self.mixup = SeparateMultiMixup(mix_beta=model_cfg.mix_beta)

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
        
        if self.both_mixup:
            if random.uniform(0, 1) > 0.5:
                self.do_mixup = True
                self.do_manifold_mixup = False
            else:
                self.do_manifold_mixup = True
                self.do_mixup = False

        if self.training and self.do_mixup:
            mixed_inputs = self.mixup(
                x=images,
                cls_labels=inputs['labels'],
                reg_labels=inputs['off_labels'],
                cls_masks=inputs['masks'],
                reg_masks=inputs['off_masks'],
            )
            images = mixed_inputs['x']
            
        b, t, c, h, w = images.shape
        x = self.model(images.view(b*t, c, h, w))  # b*t, 1280
        x = x.reshape(-1, t, self.n_channels)  # b, t, c
        x = x.permute([0, 2, 1])  # b, c, t

        if self.training and self.do_manifold_mixup:
            mixed_inputs = self.mixup(
                x=x,
                cls_labels=inputs['labels'],
                reg_labels=inputs['off_labels'],
                cls_masks=inputs['masks'],
                reg_masks=inputs['off_masks'],
            )
            x = mixed_inputs['x']
            
        if self.drop_rate > 0:
            x = self.dropout(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        cls_logits = self.cls(x)
        reg_logits = self.reg(x)
        outputs = {
            'cls': cls_logits,
            'reg': reg_logits
        }
        if self.training and (self.do_mixup or self.do_manifold_mixup):
            mixed_inputs.pop('x')
            outputs.update(mixed_inputs)
        return outputs
    
    def get_loss(self, outputs, inputs):
        if self.training and (self.do_mixup or self.do_manifold_mixup):
            cls_loss1 = self.cls_loss(outputs['cls'], outputs['cls_labels1'], outputs['cls_masks1'], outputs['coeffs'])
            cls_loss2 = self.cls_loss(outputs['cls'], outputs['cls_labels2'], outputs['cls_masks2'], 1 - outputs['coeffs'])
            cls_loss = cls_loss1 + cls_loss2
            reg_loss1 = self.reg_loss(outputs['reg'], outputs['reg_labels1'], outputs['reg_masks1'], outputs['coeffs'])
            reg_loss2 = self.reg_loss(outputs['reg'], outputs['reg_labels2'], outputs['reg_masks2'], 1 - outputs['coeffs'])
            reg_loss = reg_loss1 + reg_loss2
        else:
            cls_labels = inputs['labels']
            cls_loss = self.cls_loss(outputs['cls'], cls_labels, inputs['masks'])
            reg_loss = self.reg_loss(outputs['reg'], inputs['off_labels'], inputs['off_masks'])
        
        loss = cls_loss * self.cls_weight + reg_loss * self.reg_weight
        loss_dict = {
            'loss': loss,
            'cls': cls_loss,
            'reg': reg_loss,
        }
        return loss_dict
       
    def freeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                print(f'freeze {name}')
                for param in child.parameters():
                    param.requires_grad = False

    def unfreeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = True
        
    def freeze_bn(self, mode=True):
        count = 0
        if self._enable_pbn and mode:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
    
    
class ImageUNetLNMixupLSTM(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.n_channels = model_cfg.in_channels
        self.n_classes = model_cfg.num_classes
        bilinear = False
        self.duration = model_cfg.duration
        scale_factor = model_cfg.scale_factor
        self.drop_rate = model_cfg.drop_rate
        cls_loss_type = model_cfg.cls_loss_type
        self.cls_weight = model_cfg.cls_weight
        self.reg_weight = model_cfg.reg_weight
        self.model_name = model_cfg.model_name
        self.model_cfg = model_cfg
        
        self.model = timm.create_model(
            model_cfg.model_name,
            pretrained=model_cfg.pretrained,
            num_classes=0,
            drop_rate=model_cfg.drop,
            drop_path_rate=model_cfg.drop_path,
            # drop_block_rate=model_cfg.drop_block,
            )
            
        if model_cfg.grad_checkpointing:
            import types

            from timm.models.helpers import checkpoint_seq
            def forward_features(self, x):
                x = self.conv_stem(x)
                x = self.bn1(x)
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint_seq(self.blocks, x, flatten=False)
                else:
                    x = self.blocks(x)
                x = self.conv_head(x)
                x = self.bn2(x)
                return x
            funcType = types.MethodType
            self.model.forward_features = funcType(forward_features, self.model)
        
        if model_cfg.temporal_shift:
            self._convert_temporal_shift(model_cfg.n_div)
        if model_cfg.pretrained_path is not None:
            print(f'load backbone from {model_cfg.pretrained_path}')
            load_wraped_model_ckpt(self.model, model_cfg.pretrained_path)
            # ckpt = torch.load(model_cfg.pretrained_path, map_location='cpu')['model']
            # self.model.load_state_dict(ckpt, strict=False)

        if model_cfg.freeze_backbone:
            self.freeze_weights(freeze=['model'])
        
        if model_cfg.norm_type != 'ln':
            raise NotImplementedError()

        def create_layer_norm(channel, length):
            return nn.LayerNorm([channel, length])

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(self.n_channels, 64, norm=partial(create_layer_norm, length=self.duration))
        self.down1 = Down(64, 128, scale_factor, norm=partial(create_layer_norm, length=self.duration//2))
        self.down2 = Down(128, 256, scale_factor, norm=partial(create_layer_norm, length=self.duration//4))
        self.down3 = Down(256, 512, scale_factor, norm=partial(create_layer_norm, length=self.duration//8))
        self.down4 = Down(512, 1024 // factor, scale_factor, norm=partial(create_layer_norm, length=self.duration//16))
        self.up1 = Up(1024, 512 // factor, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration//8))
        self.up2 = Up(512, 256 // factor, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration//4))
        self.up3 = Up(256, 128 // factor, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration//2))
        self.up4 = Up(128, 64, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration))        
    
        self.gru = nn.GRU(64,64, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(128, 64, bidirectional=True, batch_first=True)
        
        self.cls = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, self.n_classes, kernel_size=1, padding=0),
            )
        self.reg = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, self.n_classes, kernel_size=1, padding=0),
            )

        if self.drop_rate > 0:
            print(f'turn on dropout with rate={self.drop_rate}')
            self.dropout = nn.Dropout(self.drop_rate)

        self.cls_loss = get_cls_loss(model_cfg)
        self.reg_loss = get_reg_loss(model_cfg)
    
        self.do_mixup = False
        self.do_manifold_mixup = False
        self.both_mixup = False
        if model_cfg.mix_beta > 0:
            if model_cfg.manifold_mixup:
                self.do_manifold_mixup = True
            else:
                self.do_mixup = True
            self.both_mixup = model_cfg.both_mixup
            self.mixup = SeparateMultiMixup(mix_beta=model_cfg.mix_beta)

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
        
        if self.both_mixup:
            if random.uniform(0, 1) > 0.5:
                self.do_mixup = True
                self.do_manifold_mixup = False
            else:
                self.do_manifold_mixup = True
                self.do_mixup = False

        if self.training and self.do_mixup:
            mixed_inputs = self.mixup(
                x=images,
                cls_labels=inputs['labels'],
                reg_labels=inputs['off_labels'],
                cls_masks=inputs['masks'],
                reg_masks=inputs['off_masks'],
            )
            images = mixed_inputs['x']
            
        b, t, c, h, w = images.shape
        x = self.model(images.view(b*t, c, h, w))  # b*t, 1280
        x = x.reshape(-1, t, self.n_channels)  # b, t, c
        x = x.permute([0, 2, 1])  # b, c, t

        if self.training and self.do_manifold_mixup:
            mixed_inputs = self.mixup(
                x=x,
                cls_labels=inputs['labels'],
                reg_labels=inputs['off_labels'],
                cls_masks=inputs['masks'],
                reg_masks=inputs['off_masks'],
            )
            x = mixed_inputs['x']
            
        if self.drop_rate > 0:
            x = self.dropout(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = x.permute([0, 2, 1])  # b, t, c
        x, _ = self.gru(x)
        x, _ = self.lstm(x)
        x = x.permute([0, 2, 1])  # b, c, t
        cls_logits = self.cls(x)
        reg_logits = self.reg(x)
        outputs = {
            'cls': cls_logits,
            'reg': reg_logits
        }
        if self.training and (self.do_mixup or self.do_manifold_mixup):
            mixed_inputs.pop('x')
            outputs.update(mixed_inputs)
        return outputs
    
    def get_loss(self, outputs, inputs):
        if self.training and (self.do_mixup or self.do_manifold_mixup):
            cls_loss1 = self.cls_loss(outputs['cls'], outputs['cls_labels1'], outputs['cls_masks1'], outputs['coeffs'])
            cls_loss2 = self.cls_loss(outputs['cls'], outputs['cls_labels2'], outputs['cls_masks2'], 1 - outputs['coeffs'])
            cls_loss = cls_loss1 + cls_loss2
            reg_loss1 = self.reg_loss(outputs['reg'], outputs['reg_labels1'], outputs['reg_masks1'], outputs['coeffs'])
            reg_loss2 = self.reg_loss(outputs['reg'], outputs['reg_labels2'], outputs['reg_masks2'], 1 - outputs['coeffs'])
            reg_loss = reg_loss1 + reg_loss2
        else:
            cls_labels = inputs['labels']
            cls_loss = self.cls_loss(outputs['cls'], cls_labels, inputs['masks'])
            reg_loss = self.reg_loss(outputs['reg'], inputs['off_labels'], inputs['off_masks'])
        
        loss = cls_loss * self.cls_weight + reg_loss * self.reg_weight
        loss_dict = {
            'loss': loss,
            'cls': cls_loss,
            'reg': reg_loss,
        }
        return loss_dict
       
    def freeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                print(f'freeze {name}')
                for param in child.parameters():
                    param.requires_grad = False

    def unfreeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = True
        
    def freeze_bn(self, mode=True):
        count = 0
        if self._enable_pbn and mode:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                                             

class ShallowImageUNetLNMixup(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.n_channels = model_cfg.in_channels
        self.n_classes = model_cfg.num_classes
        bilinear = False
        self.duration = model_cfg.duration
        scale_factor = model_cfg.scale_factor
        self.drop_rate = model_cfg.drop_rate
        cls_loss_type = model_cfg.cls_loss_type
        self.cls_weight = model_cfg.cls_weight
        self.reg_weight = model_cfg.reg_weight
        self.model_name = model_cfg.model_name
        self.model_cfg = model_cfg
        
        self.model = timm.create_model(
            model_cfg.model_name,
            pretrained=model_cfg.pretrained,
            num_classes=0,
            drop_rate=model_cfg.drop,
            drop_path_rate=model_cfg.drop_path,
            # drop_block_rate=model_cfg.drop_block,
            )
            
        if model_cfg.grad_checkpointing:
            import types

            from timm.models.helpers import checkpoint_seq
            def forward_features(self, x):
                x = self.conv_stem(x)
                x = self.bn1(x)
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint_seq(self.blocks, x, flatten=False)
                else:
                    x = self.blocks(x)
                x = self.conv_head(x)
                x = self.bn2(x)
                return x
            funcType = types.MethodType
            self.model.forward_features = funcType(forward_features, self.model)
        
        if model_cfg.temporal_shift:
            self._convert_temporal_shift(model_cfg.n_div)
        if model_cfg.pretrained_path is not None:
            print(f'load backbone from {model_cfg.pretrained_path}')
            load_wraped_model_ckpt(self.model, model_cfg.pretrained_path)
            # ckpt = torch.load(model_cfg.pretrained_path, map_location='cpu')['model']
            # self.model.load_state_dict(ckpt, strict=False)

        if model_cfg.freeze_backbone:
            self.freeze_weights(freeze=['model'])
        
        if model_cfg.norm_type != 'ln':
            raise NotImplementedError()

        def create_layer_norm(channel, length):
            return nn.LayerNorm([channel, length])

        self.inc = DoubleConv(self.n_channels, 128, norm=partial(create_layer_norm, length=self.duration), se=model_cfg.se, res=model_cfg.res)
        self.down1 = Down(128, 256, scale_factor, norm=partial(create_layer_norm, length=self.duration//2), se=model_cfg.se, res=model_cfg.res)
        self.down2 = Down(256, 512, scale_factor, norm=partial(create_layer_norm, length=self.duration//4), se=model_cfg.se, res=model_cfg.res)
        factor = 2 if bilinear else 1
        self.up1 = Up(512, 256 // factor, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration//2))
        self.up2 = Up(256, 128 // factor, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration))
        self.cls = OutConv(128, self.n_classes, kernel_size=1, padding=0)
        self.reg = OutConv(128, self.n_classes, kernel_size=1, padding=0)
        
        if self.drop_rate > 0:
            print(f'turn on dropout with rate={self.drop_rate}')
            self.dropout = nn.Dropout(self.drop_rate)

        self.cls_loss = get_cls_loss(model_cfg)
        self.reg_loss = get_reg_loss(model_cfg)
    
        self.do_mixup = False
        self.do_manifold_mixup = False
        assert model_cfg.mix_beta > 0
        if model_cfg.manifold_mixup:
            self.do_manifold_mixup = True
        else:
            self.do_mixup = True
        self.both_mixup = model_cfg.both_mixup
        self.mixup = SeparateMultiMixup(mix_beta=model_cfg.mix_beta)

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
        
        if self.both_mixup:
            if random.uniform(0, 1) > 0.5:
                self.do_mixup = True
                self.do_manifold_mixup = False
            else:
                self.do_manifold_mixup = True
                self.do_mixup = False

        if self.training and self.do_mixup:
            mixed_inputs = self.mixup(
                x=images,
                cls_labels=inputs['labels'],
                reg_labels=inputs['off_labels'],
                cls_masks=inputs['masks'],
                reg_masks=inputs['off_masks'],
            )
            images = mixed_inputs['x']
            
        b, t, c, h, w = images.shape
        x = self.model(images.view(b*t, c, h, w))  # b*t, 1280
        x = x.reshape(-1, t, self.n_channels)  # b, t, c
        x = x.permute([0, 2, 1])  # b, c, t

        if self.training and self.do_manifold_mixup:
            mixed_inputs = self.mixup(
                x=x,
                cls_labels=inputs['labels'],
                reg_labels=inputs['off_labels'],
                cls_masks=inputs['masks'],
                reg_masks=inputs['off_masks'],
            )
            x = mixed_inputs['x']
            
        if self.drop_rate > 0:
            x = self.dropout(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        cls_logits = self.cls(x)
        reg_logits = self.reg(x)
        outputs = {
            'cls': cls_logits,
            'reg': reg_logits
        }
        if self.training and (self.do_mixup or self.do_manifold_mixup):
            mixed_inputs.pop('x')
            outputs.update(mixed_inputs)
        return outputs
    
    def get_loss(self, outputs, inputs):
        if self.training and (self.do_mixup or self.do_manifold_mixup):
            cls_loss1 = self.cls_loss(outputs['cls'], outputs['cls_labels1'], outputs['cls_masks1'], outputs['coeffs'])
            cls_loss2 = self.cls_loss(outputs['cls'], outputs['cls_labels2'], outputs['cls_masks2'], 1 - outputs['coeffs'])
            cls_loss = cls_loss1 + cls_loss2
            reg_loss1 = self.reg_loss(outputs['reg'], outputs['reg_labels1'], outputs['reg_masks1'], outputs['coeffs'])
            reg_loss2 = self.reg_loss(outputs['reg'], outputs['reg_labels2'], outputs['reg_masks2'], 1 - outputs['coeffs'])
            reg_loss = reg_loss1 + reg_loss2
        else:
            cls_labels = inputs['labels']
            cls_loss = self.cls_loss(outputs['cls'], cls_labels, inputs['masks'])
            reg_loss = self.reg_loss(outputs['reg'], inputs['off_labels'], inputs['off_masks'])
        
        loss = cls_loss * self.cls_weight + reg_loss * self.reg_weight
        loss_dict = {
            'loss': loss,
            'cls': cls_loss,
            'reg': reg_loss,
        }
        return loss_dict
       
    def freeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                print(f'freeze {name}')
                for param in child.parameters():
                    param.requires_grad = False

    def unfreeze_weights(self, freeze=[]):
        for name, child in self.named_children():
            if name in freeze:
                for param in child.parameters():
                    param.requires_grad = True
        
    def freeze_bn(self, mode=True):
        count = 0
        if self._enable_pbn and mode:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
   