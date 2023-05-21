#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from tqdm import trange
from pathlib import Path

from styletransfer.utils import get_losses, tensor_to_img

def main(args):
  device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  cnn = torchvision.models.vgg19(weights="IMAGENET1K_V1").features.to(device).eval()
  for i, m in enumerate(cnn):
    if isinstance(m, nn.MaxPool2d):
      cnn[i] = nn.AvgPool2d(kernel_size=m.kernel_size, stride=m.stride, padding=m.padding)
    elif isinstance(m, nn.ReLU):
      cnn[i] = nn.ReLU(inplace=False)

  image_size = 512 if args.gpu else 128
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)
  transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
  ])

  content = Image.open(args.content)
  style = Image.open(args.style)

  content = transform(content).unsqueeze(0).to(device)
  style = transform(style).unsqueeze(0).to(device)

  content_layers = [int(x) for x in args.content_layers.split(",")]
  style_layers = [int(x) for x in args.style_layers.split(",")]
  layers_needed = list(set(content_layers + style_layers))

  cnn = cnn[:max(layers_needed)+1]
  cnn.requires_grad_(False)

  if args.init == "random":
    gen = torch.randn_like(content).requires_grad_(True).to(device)
  else:
    gen = content.clone().detach().requires_grad_(True).to(device)
  optim = torch.optim.Adam([gen], lr=args.lr)
  sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[int(0.5*args.steps), int(0.85*args.steps)], gamma=0.1)

  content_losses, style_losses, tv_loss = get_losses(cnn, content, style, content_layers, style_layers)

  for s in (t := trange(args.steps)):
    optim.zero_grad(set_to_none=True)

    content_loss = 0
    style_loss = 0
    f = gen
    for i, layer in enumerate(cnn):
      f = layer(f)
      if i in content_layers:
        content_loss += content_losses[i](f)
      if i in style_layers:
        style_loss += (1/len(style_layers)) * style_losses[i](f)
    tv = tv_loss(gen)
    loss = args.content_weight * content_loss + args.style_weight * style_loss + args.tv_weight * tv
    loss.backward()
    optim.step()
    sched.step()

    t.set_description(f"Steps {s}: {loss.item():.4f} - Content: {content_loss.item():.4f} - Style: {style_loss.item():.4f} - TV: {tv.item():.4f}")
    if args.save_each != 0 and s % args.save_each == 0:
      img = tensor_to_img(gen, mean, std)
      img.save(Path(args.out) / f"{s}.jpg")
  img = tensor_to_img(gen, mean, std)
  img.save(Path(args.out) / "final.jpg")
  print(f"Saved final image to {Path(args.out) / 'final.jpg'}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("content", type=str)
  parser.add_argument("style", type=str)
  parser.add_argument("--content_layers", type=str, default="21")
  parser.add_argument("--style_layers", type=str, default="0,5,10,19,28")
  parser.add_argument("--init", type=str, default="random")
  parser.add_argument("--lr", type=float, default=1e-1)
  parser.add_argument("--content_weight", type=float, default=5)
  parser.add_argument("--style_weight", type=float, default=1e5)
  parser.add_argument("--tv_weight", type=float, default=2)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--save_each", type=int, default=0)
  parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument("--out", type=str, default="out/img")
  args = parser.parse_args()
  main(args)
