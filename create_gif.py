#!/usr/bin/env python

import os
import imageio
import re
import argparse

from pathlib import Path

def extract_number(filename):
  # Extracts the numeric portion from a filename
  match = re.search(r'\d+', filename)
  if match:
    return int(match.group())
  return -1

def create_gif(image_dir, gif_dir):
  image_dir = Path(image_dir)
  gif_dir = Path(gif_dir)

  images = []
  image_files = [d.name for d in image_dir.iterdir() if d.is_file() and d.suffix in [".png", ".jpg", ".jpeg"]]
  image_files = sorted(image_files, key=extract_number)
  for image_file in image_files:
    if "final" in image_file: continue
    image_path = image_dir / image_file
    images.append(imageio.imread(image_path))

  # find first filename available
  i = 0
  while (gif_dir / f"{i}.gif").exists():
    i += 1

  filename = f"{i}.gif"
  imageio.mimsave(os.path.join(gif_dir, filename), images)
  print(f"Saved gif to {os.path.join(gif_dir, filename)}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--image_dir", type=str, default="out/img")
  parser.add_argument("--gif_dir", type=str, default="out/gif")
  args = parser.parse_args()
  create_gif(args.image_dir, args.gif_dir)
