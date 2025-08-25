#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F
from torchvision.models import vgg19
from torch.nn import BCEWithLogitsLoss

class EnhancedHybridLoss(torch.nn.Module):
    def __init__(self, alpha=1e-3, beta=1e-3, gamma=2, lambda_adv=1e-2, 
                 spatial_tv=True, spectral_tv=True, perceptual_layers=[3, 8, 15]):
        super(EnhancedHybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_adv = lambda_adv
        self.vgg = vgg19(pretrained=True).features.eval()
        self.perceptual_layers = perceptual_layers
        self.focal_loss = BCEWithLogitsLoss(reduction='none')
        self.spatial_tv = TVLoss() if spatial_tv else None
        self.spectral_tv = TVLossSpectral() if spectral_tv else None

    def perceptual(self, x, y):
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x, y = layer(x), layer(y)
            if i in self.perceptual_layers:
                loss += F.mse_loss(x, y)
        return loss

    def adversarial(self, logits, is_real):
        targets = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
        focal_weights = self.alpha * torch.pow(targets - torch.sigmoid(logits), self.gamma)
        loss = focal_weights * self.focal_loss(logits, targets)
        return loss.mean()

    def forward(self, y, gt, logits_real=None, logits_fake=None):
        l1_loss = F.l1_loss(y, gt)

        spatial_tv = self.spatial_tv(y) if self.spatial_tv else 0.0
        spectral_tv = self.spectral_tv(y) if self.spectral_tv else 0.0
        perceptual_loss = self.beta * self.perceptual(y, gt)

        adversarial_loss = 0.0
        if logits_real is not None and logits_fake is not None:
            adversarial_loss = self.adversarial(logits_real, True) + self.adversarial(logits_fake, False)
            adversarial_loss *= self.lambda_adv

        total_loss = l1_loss + spatial_tv + spectral_tv + perceptual_loss + adversarial_loss
        return total_loss

