import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterNetLoss(nn.Module):
    # from https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, pad_mask: torch.Tensor):
        y_pred = y_pred.float()
        y_true = y_true.float()

        # y_pred = y_pred.sigmoid()
        y_pred = torch.clamp(y_pred.sigmoid_(), min=1e-5, max=1 - 1e-5)

        pad_mask = pad_mask.bool()
        if pad_mask.sum() == 0:
            return torch.tensor(0.0).to(y_pred)
        y_pred = y_pred[pad_mask]
        y_true = y_true[pad_mask]

        pos_inds = y_true.eq(1).float()
        neg_inds = y_true.lt(1).float()

        neg_weights = torch.pow(1 - y_true, 4)

        loss = 0

        pos_loss = torch.log(y_pred) * torch.pow(1 - y_pred, 2) * pos_inds
        neg_loss = torch.log(1 - y_pred) * torch.pow(y_pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, num_classes, beta=0.9999, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.weights = self._calc_weight(samples_per_class, num_classes, beta)

    def _calc_weight(self, samples_per_class, num_classes, beta):
        effective_num = 1.0 - torch.pow(beta, torch.tensor(samples_per_class))
        weights = (1.0 - beta) / effective_num
        weights = weights / torch.sum(weights)
        weights = weights.unsqueeze(0).float()
        return weights.cuda()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, pad_mask: torch.Tensor, mixup_coeffs=None):
        y_pred = y_pred.permute([0, 2, 1]).reshape(-1, self.num_classes)
        y_true = y_true.permute([0, 2, 1]).reshape(-1, self.num_classes)

        if mixup_coeffs is not None:
            mixup_coeffs = mixup_coeffs[:, None, None].expand_as(pad_mask)
            mixup_coeffs = mixup_coeffs.permute([0, 2, 1]).reshape(-1, self.num_classes)

        ##############################
        # just don't use mask.
        # treat outside of mask as no event at all.
        ##############################
        # pad_mask = pad_mask.permute([0, 2, 1]).reshape(-1, self.num_classes)

        # pad_mask = pad_mask[:, 0].bool()
        # if pad_mask.sum() == 0:
        #     return torch.tensor(0.0).to(y_pred)
        # y_pred = y_pred[pad_mask]
        # y_true = y_true[pad_mask]

        p = torch.sigmoid(y_pred)
        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        p_t = p * y_true + (1 - p) * (1 - y_true)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        alpha = self.weights
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = alpha_t * loss

        if mixup_coeffs is not None:
            loss = loss * mixup_coeffs

        return loss.mean()


class MaskedClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, num_classes, beta=0.9999, gamma=2.0, alpha=0.25, manual_weights=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        self.weights = self._calc_weight(samples_per_class, num_classes, beta)
        if manual_weights is not None:
            self.manual_weights = torch.tensor(manual_weights)
        else:
            self.manual_weights = manual_weights

    def _calc_weight(self, samples_per_class, num_classes, beta):
        effective_num = 1.0 - torch.pow(beta, torch.tensor(samples_per_class))
        weights = (1.0 - beta) / effective_num
        weights = weights / torch.sum(weights)
        weights = weights.unsqueeze(0).float()
        return weights.cuda()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, pad_mask: torch.Tensor, mixup_coeffs=None):
        y_pred = y_pred.permute([0, 2, 1]).reshape(-1, self.num_classes)
        y_true = y_true.permute([0, 2, 1]).reshape(-1, self.num_classes)

        if mixup_coeffs is not None:
            mixup_coeffs = mixup_coeffs[:, None, None].expand_as(pad_mask)
            mixup_coeffs = mixup_coeffs.permute([0, 2, 1]).reshape(-1, self.num_classes)

        pad_mask = pad_mask.permute([0, 2, 1]).reshape(-1, self.num_classes)

        pad_mask = pad_mask[:, 0].bool()
        if pad_mask.sum() == 0:
            return torch.tensor(0.0).to(y_pred)
        y_pred = y_pred[pad_mask]
        y_true = y_true[pad_mask]

        p = torch.sigmoid(y_pred)
        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        p_t = p * y_true + (1 - p) * (1 - y_true)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        alpha = self.weights
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = alpha_t * loss

        if mixup_coeffs is not None:
            mixup_coeffs = mixup_coeffs[pad_mask]
            loss = loss * mixup_coeffs

        if self.manual_weights is not None:
            loss *= self.manual_weights.to(loss)
            loss = loss.sum(dim=1)

        return loss.mean()


class MaskedHuberLoss(nn.Module):
    def __init__(self, manual_weights=None):
        super().__init__()
        self.num_classes = 3
        if manual_weights is not None:
            self.manual_weights = torch.tensor(manual_weights)
        else:
            self.manual_weights = manual_weights

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, pad_mask: torch.Tensor, mixup_coeffs=None):
        y_pred = y_pred.permute([0, 2, 1]).reshape(-1, self.num_classes)
        y_true = y_true.permute([0, 2, 1]).reshape(-1, self.num_classes)

        if mixup_coeffs is not None:
            mixup_coeffs = mixup_coeffs[:, None, None].expand_as(pad_mask)
            mixup_coeffs = mixup_coeffs.permute([0, 2, 1]).reshape(-1, self.num_classes)

        pad_mask = pad_mask.permute([0, 2, 1]).reshape(-1, self.num_classes)

        pad_mask = pad_mask[:, 0].bool()
        if pad_mask.sum() == 0:
            return torch.tensor(0.0).to(y_pred)
        y_pred = y_pred[pad_mask]
        y_true = y_true[pad_mask]

        loss = torch.nn.functional.smooth_l1_loss(y_pred, y_true, reduction='none')

        if mixup_coeffs is not None:
            mixup_coeffs = mixup_coeffs[pad_mask]
            loss = loss * mixup_coeffs

        if self.manual_weights is not None:
            loss *= self.manual_weights.to(loss)
            loss = loss.sum(dim=1)

        return loss.mean()


def get_cls_loss(model_cfg):
    cls_loss_type = model_cfg.cls_loss_type
    if cls_loss_type == 'cb_focal':
        cls_loss = MaskedClassBalancedFocalLoss(
            samples_per_class=model_cfg.samples_per_class,
            num_classes=model_cfg.num_classes,
            beta=model_cfg.beta,
            gamma=model_cfg.gamma,
            alpha=model_cfg.alpha,
            manual_weights=model_cfg.manual_weights)
    elif cls_loss_type == 'no_mask_cb_focal':
        cls_loss = ClassBalancedFocalLoss(
            samples_per_class=model_cfg.samples_per_class,
            num_classes=model_cfg.num_classes,
            beta=model_cfg.beta,
            gamma=model_cfg.gamma,
            alpha=model_cfg.alpha)
    else:
        cls_loss = None
    return cls_loss


def get_reg_loss(model_cfg):
    reg_loss = MaskedHuberLoss(manual_weights=model_cfg.manual_weights)
    return reg_loss
