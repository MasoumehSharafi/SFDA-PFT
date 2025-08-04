# --- src/losses.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F


def emotion_loss(pred1, pred2):
    return nn.MSELoss()(pred1, pred2)

def emotion_loss_ce(pred1, pred2):
    return nn.CrossEntropyLoss()(pred1, pred2)

def emotion_loss_kl(pred1, pred2):
    log_prob = F.log_softmax(pred1, dim=1)  # log-probabilities
    return F.kl_div(log_prob, pred2, reduction='batchmean')

def get_emotion_loss_fn(loss_type='mse'):
    if loss_type == 'mse':
        return lambda p1, p2: nn.MSELoss()(p1, p2)
    elif loss_type == 'ce':
        return lambda p1, p2: nn.CrossEntropyLoss()(p1, p2.argmax(dim=1))  # assumes p2 is one-hot
    elif loss_type == 'kl':
        return lambda p1, p2: F.kl_div(F.log_softmax(p1, dim=1), p2, reduction='batchmean')
    else:
        raise ValueError(f"Unsupported emotion loss type: {loss_type}")


def stats(f):
    # f shape: [batch, feature_dim] or [batch, channels, h, w]
    if f.dim() == 4:
        mu = f.mean(dim=[2, 3])  # spatial mean
        sigma = f.var(dim=[2, 3])
    elif f.dim() == 2:
        mu = f.mean(dim=1, keepdim=True)  # mean over feature dim
        sigma = f.var(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unexpected feature dimension: {f.shape}")
    return mu, sigma

def style_loss(f1, f2):
    mu1, sigma1 = stats(f1)
    mu2, sigma2 = stats(f2)
    return ((mu1 - mu2)**2).mean() + ((sigma1 - sigma2)**2).mean()

def moment_loss(feat_orig, feat_trans):
    mu_orig = feat_orig.mean(dim=0)
    mu_trans = feat_trans.mean(dim=0)
    std_orig = feat_orig.std(dim=0)
    std_trans = feat_trans.std(dim=0)
    return ((mu_orig - mu_trans) ** 2).mean() + ((std_orig - std_trans) ** 2).mean()
