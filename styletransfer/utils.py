import numpy as np
from PIL import Image

from .loss import ContentLoss, StyleLoss, TVLoss

def get_losses(model, content, style, content_layers, style_layers):
  content_losses = {}
  style_losses = {}
  tv_loss = TVLoss()
  for i, layer in enumerate(model):
    content = layer(content)
    style = layer(style)
    if i in content_layers:
      content_losses[i] = ContentLoss(content)
    if i in style_layers:
      style_losses[i] = StyleLoss(style)
  return content_losses, style_losses, tv_loss

def tensor_to_img(tensor, mean, std):
  image = tensor.to("cpu").clone().detach()
  image = image.numpy().squeeze()
  image = image.transpose(1, 2, 0)
  image = image * np.array(std) + np.array(mean)
  image = image.clip(0, 1)
  return Image.fromarray((image * 255).astype(np.uint8))
