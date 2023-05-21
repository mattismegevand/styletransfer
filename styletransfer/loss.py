import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
  def __init__(self, target):
    super(ContentLoss, self).__init__()
    self.target = target.detach()

  def forward(self, input):
    return F.mse_loss(input, self.target)

class StyleLoss(nn.Module):
  def __init__(self, target):
    super(StyleLoss, self).__init__()
    self.target = self.gramm_matrix(target).detach()

  def forward(self, input):
    return F.mse_loss(self.gramm_matrix(input), self.target)

  @staticmethod
  def gramm_matrix(input):
    a, b, c, d = input.shape
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class TVLoss(nn.Module):
  def __init__(self):
    super(TVLoss, self).__init__()

  def forward(self, input):
    _, _, c, d = input.shape
    loss = F.mse_loss(input[:, :, 1:, :], input[:, :, :c - 1, :]) + \
           F.mse_loss(input[:, :, :, 1:], input[:, :, :, :d - 1])
    return loss
