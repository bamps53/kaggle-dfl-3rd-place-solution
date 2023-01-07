import torch

def get_optimizer(model, cfg):
    def exclude(
        n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

    def include(n, p): return not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(
        n, p) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": cfg.wd},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    return optimizer
