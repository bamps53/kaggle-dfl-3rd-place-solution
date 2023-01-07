import torch
from torch import nn
from torch.distributions import Beta


class MultiMixup(nn.Module):
    def __init__(self, mix_beta=0.5, return_perm=False):

        super().__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.return_perm = return_perm

    @staticmethod
    def mix(x, coeffs, perm):
        n_dims = len(x.shape)
        if n_dims == 2:
            x = coeffs.view(-1, 1) * x + (1 - coeffs.view(-1, 1)) * x[perm]
        elif n_dims == 3:
            x = coeffs.view(-1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1)) * x[perm]
        elif n_dims == 4:
            x = coeffs.view(-1, 1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1, 1)) * x[perm]
        else:
            x = coeffs.view(-1, 1, 1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1, 1, 1)) * x[perm]
        return x

    def forward(self, X, Y1, Y2):
        bs = X.shape[0]
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)
        
        X = self.mix(X, coeffs, perm)
        Y1 = self.mix(Y1, coeffs, perm)
        Y2 = self.mix(Y2, coeffs, perm)

        if self.return_perm:
            return X, Y1, Y2, perm, coeffs
        return X, Y1, Y2
    
class SeparateMultiMixup(nn.Module):
    def __init__(self, mix_beta=0.5, return_perm=False):

        super().__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.return_perm = return_perm

    @staticmethod
    def mix(x, coeffs, perm):
        n_dims = len(x.shape)
        if n_dims == 2:
            x = coeffs.view(-1, 1) * x + (1 - coeffs.view(-1, 1)) * x[perm]
        elif n_dims == 3:
            x = coeffs.view(-1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1)) * x[perm]
        elif n_dims == 4:
            x = coeffs.view(-1, 1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1, 1)) * x[perm]
        else:
            x = coeffs.view(-1, 1, 1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1, 1, 1)) * x[perm]
        return x

    def forward(self, x, cls_labels, reg_labels, cls_masks, reg_masks):
        bs = x.shape[0]
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(x.device)
        
        x = self.mix(x, coeffs, perm)
        
        cls_labels1, cls_labels2 = cls_labels, cls_labels[perm]
        reg_labels1, reg_labels2 = reg_labels, reg_labels[perm]
        
        cls_masks1, cls_masks2 = cls_masks, cls_masks[perm]
        reg_masks1, reg_masks2 = reg_masks, reg_masks[perm]
        
        outputs = {}
        outputs['x'] = x
        outputs['cls_labels1'] = cls_labels1
        outputs['cls_labels2'] = cls_labels2
        outputs['reg_labels1'] = reg_labels1
        outputs['reg_labels2'] = reg_labels2
        outputs['cls_masks1'] = cls_masks1
        outputs['cls_masks2'] = cls_masks2
        outputs['reg_masks1'] = reg_masks1
        outputs['reg_masks2'] = reg_masks2

        outputs['coeffs'] = coeffs
        outputs['perm'] = perm
        
        return outputs
    
class ClsOnlySeparateMultiMixup(nn.Module):
    def __init__(self, mix_beta=0.5, return_perm=False):

        super().__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.return_perm = return_perm

    @staticmethod
    def mix(x, coeffs, perm):
        n_dims = len(x.shape)
        if n_dims == 2:
            x = coeffs.view(-1, 1) * x + (1 - coeffs.view(-1, 1)) * x[perm]
        elif n_dims == 3:
            x = coeffs.view(-1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1)) * x[perm]
        elif n_dims == 4:
            x = coeffs.view(-1, 1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1, 1)) * x[perm]
        else:
            x = coeffs.view(-1, 1, 1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1, 1, 1)) * x[perm]
        return x

    def forward(self, x, cls_labels, cls_masks):
        bs = x.shape[0]
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(x.device)
        
        x = self.mix(x, coeffs, perm)
        
        cls_labels1, cls_labels2 = cls_labels, cls_labels[perm]
        cls_masks1, cls_masks2 = cls_masks, cls_masks[perm]
        
        outputs = {}
        outputs['x'] = x
        outputs['cls_labels1'] = cls_labels1
        outputs['cls_labels2'] = cls_labels2
        outputs['cls_masks1'] = cls_masks1
        outputs['cls_masks2'] = cls_masks2
        outputs['coeffs'] = coeffs
        outputs['perm'] = perm
        
        return outputs