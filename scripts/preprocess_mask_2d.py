import pandas as pd
import numpy as np
import cv2
import nibabel as nib
import os
from tqdm import tqdm
import multiprocessing as mp

ROOT = "./input/rsna-2022-cervical-spine-fracture-detection/"

OUTFOLDER = ROOT + "masks_2d/"
SEGFOLDER = ROOT + "segmentations/"
TRAIN_IMAGES_FOLDER = ROOT + "train_images/"


reversed_dicoms = pd.read_csv(ROOT + "train_folded_v1.csv")
reversed_dicoms = reversed_dicoms[reversed_dicoms["is_reversed"] > 0][
    "StudyInstanceUID"
].unique()


seg_labels = os.listdir(SEGFOLDER)
len(seg_labels), seg_labels[:5]


def load_mask_3d(fp):
    img = nib.load(fp)
    img = img.get_fdata()  # convert to numpy array
    img = img[:, ::-1, ::-1].transpose(2, 1, 0)  # align orientation with train image
    return img


def do_one(fn):
    study_id = fn[:-4]
    mask3d = load_mask_3d(SEGFOLDER + fn)

    if study_id in reversed_dicoms:
        mask3d = mask3d[::-1]

    dicoms = os.listdir(f"{TRAIN_IMAGES_FOLDER}/{study_id}/")
    dicoms = np.sort([int(item[:-4]) for item in dicoms])
    os.makedirs(f"{OUTFOLDER}/{study_id}/", exist_ok=True)
    assert len(dicoms) == len(mask3d)

    for i, mask in zip(dicoms, mask3d):
        cv2.imwrite(f"{OUTFOLDER}/{study_id}/{i}.png", mask)


with mp.Pool(16) as p:
    res = list(tqdm(p.imap(do_one, seg_labels), total=len(seg_labels)))
