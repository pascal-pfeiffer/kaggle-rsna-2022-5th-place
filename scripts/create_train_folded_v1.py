import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold


ROOT = "./input/rsna-2022-cervical-spine-fracture-detection/"

train = pd.read_csv(ROOT + "train.csv")

seg_labels = os.listdir(ROOT + "segmentations/")
seg_ids = np.array([item[:-4] for item in seg_labels])

train["has_segmentation"] = train["StudyInstanceUID"].isin(seg_ids).astype(int)

train_boxes = pd.read_csv(ROOT + "train_bounding_boxes.csv")
box_ids = train_boxes["StudyInstanceUID"].unique()

train["has_boxes"] = train["StudyInstanceUID"].isin(box_ids).astype(int)

x = pd.DataFrame(
    train[train["has_segmentation"] == 1][
        ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    ].mean(0),
    columns=["has_mask"],
)
x["no_mask"] = train[train["has_segmentation"] == 0][
    ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
].mean(0)

kf = KFold(n_splits=5, random_state=23, shuffle=True)
splits = list(kf.split(train))

train["fold"] = -1
for fold, s in enumerate(splits):
    train.iloc[s[1], -1] = fold


reversed_dicoms = pd.read_csv(ROOT + "reversed_dicoms.txt", header=None).values[:, 0]

train["is_reversed"] = train["StudyInstanceUID"].isin(reversed_dicoms).astype(int)

train.to_csv(ROOT + "train_folded_v1.csv", index=False)
