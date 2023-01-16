import numpy as np
import torch


def mixup(x, alpha, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x


def dino_mixup(images, alpha=1.0):
    return [mixup(image, alpha) for image in images]


def cutmix(x, beta):
    out = x.clone()
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(x.size()[0]).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    out[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    return out


def dino_cutmix(images, beta=1.0):
    return [cutmix(image, beta) for image in images]


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
