## Imports

from PIL import Image
from tqdm import tqdm
tqdm.pandas()
import os
import shutil
import pandas as pd
from random import random

## Constants

IMG_DIR = "base_imgs"
IMG_SAVE_DIR = "sorted_imgs"
TRANSFORMS = {
        "identity":   lambda im: im,
        "rot90":      lambda im: im.transpose(Image.ROTATE_90),
        "rot180":     lambda im: im.transpose(Image.ROTATE_180),
        "rot270":     lambda im: im.transpose(Image.ROTATE_270),
        "flip":       lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        "rot90flip":  lambda im: im.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT),
        "rot180flip": lambda im: im.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT),
        "rot270flip": lambda im: im.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)}

OPPOSITE_ANSWER = {
        "Farming":           "No Farming",
        "Restoration":       "No Restoration",
        "Sea Level Rise":    "No Sea Level Rise",
        "Shipping":          "No Shipping",
        "Oil and Gas":       "Not Oil and Gas",
        "Shoreline Erosion": "No Shoreline Erosion"}

# Make opposite answer dictionary bidirectional
for key, val in list(OPPOSITE_ANSWER.items()):
    OPPOSITE_ANSWER[val] = key

##

def handle_row(row):
    print(row.name)
    experts_say_yes = row["expert_percentage"] > 0.65
    experts_say_no  = row["expert_percentage"] < 0.35
    if not (experts_say_yes or experts_say_no): return
    with Image.open(f"{IMG_DIR}/{row['image']}.jpg") as im:
        for transform_name, transform in TRANSFORMS.items():
            training_or_validation = "val" if random() < 0.2 else "train"
            expert_answer  = None
            if experts_say_yes:
                expert_answer = row['majority_answer']
            elif experts_say_no:
                expert_answer = OPPOSITE_ANSWER[row['majority_answer']]
            else:
                raise Exception
            transform(im).save(f"{IMG_SAVE_DIR}/{training_or_validation}/{expert_answer}/{row['image']}_{transform_name}.jpg")

votes = pd.read_csv("all_votes.csv")

votes.apply(handle_row, axis=1)
