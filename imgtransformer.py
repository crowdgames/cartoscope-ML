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
IMG_SAVE_DIR = "expert_rated_imgs"

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

votes = pd.read_csv("all_votes.csv")

##

def split_by_experts(row):
    experts_agree    = row["expert_percentage"] > 0.65
    experts_disagree = row["expert_percentage"] < 0.35
    if not (experts_agree or experts_disagree): return
    # img bucket
    image_bucket = None
    bucket_rand = random()
    if bucket_rand < 0.2:
        image_bucket = "test"
    elif bucket_rand < 0.4:
        image_bucket = "val"
    else:
        image_bucket = "train"
    project = row['majority_answer'] if row['pattern_identified'] else OPPOSITE_ANSWER[row['majority_answer']]
    expert_pattern_identified = row['pattern_identified'] if experts_agree else 1 - row['pattern_identified']
    expert_answer = "yes" if expert_pattern_identified else "no"
    src  = f"{IMG_DIR}/{row['image']}.jpg"
    dest = f"{IMG_SAVE_DIR}/{project}/{image_bucket}/{expert_answer}/{row['image']}.jpg"
    shutil.copy(src, dest)

##

votes.apply(split_by_experts, axis=1)

##

def get_row_for_proj_img(project, img_name):
    of_image   = votes["image"] == img_name
    of_project = votes["majority_answer"].isin((project, OPPOSITE_ANSWER[project]))
    return votes[of_image & of_project]

##

for project in tqdm(filter(lambda s: s[0] != 'N', OPPOSITE_ANSWER.keys())):
    for img_bucket in ["train", "val"]:
        for expert_yes_no in ["yes", "no"]:
            for img_full_name in os.listdir(f"./expert_rated_imgs/{project}/{img_bucket}/{expert_yes_no}"):
                img_name    = img_full_name[:-4]
                row         = get_row_for_proj_img(project, img_name)
                assert len(row) == 1
                user_answer = "yes" if row["majority_answer"].iloc[0] == project else "no"
                src         = f"./expert_rated_imgs/{project}/{img_bucket}/{expert_yes_no}/{img_full_name}"
                dest        = f"./user_rated_imgs/{project}/{img_bucket}/{user_answer}/{img_full_name}"
                shutil.copy(src, dest)
