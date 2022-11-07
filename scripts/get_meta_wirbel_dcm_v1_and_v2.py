from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import multiprocessing as mp
import pydicom


ROOT = "./input/rsna-2022-cervical-spine-fracture-detection/"

meta = pd.read_csv(ROOT + "meta_segmentation.csv")
print(meta["ImageHeight"].unique(), meta["ImageWidth"].unique())

meta["Image"] = (
    meta["StudyInstanceUID"].astype(str) + "/" + meta["Slice"].astype(str) + ".dcm"
)

N_FOLDS = 5

split = GroupKFold(N_FOLDS)
for k, (_, test_idx) in enumerate(split.split(meta, groups=meta.StudyInstanceUID)):
    meta.loc[test_idx, "fold"] = k

meta.to_csv(ROOT + "meta_wirbel_dcm_v0.csv", index=False)

MASK_FOLDER = ROOT + "masks_2d/"

seg_labels = os.listdir(ROOT + "/segmentations/")
len(seg_labels), seg_labels[:5]


train = pd.read_csv(ROOT + "meta_wirbel_dcm_v0.csv")


def load_mask(fp):
    mask = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    return mask


targets = np.zeros((train.shape[0], 7))


for i, img_fn in tqdm(enumerate(train["Image"].values), total=len(train)):
    fn = img_fn.replace("dcm", "png")
    fp = MASK_FOLDER + fn
    mask = load_mask(fp)
    mask[mask > 7] = 0
    l = np.unique(mask)[1:] - 1
    targets[i, l] = 1


old_targets = train[["C1", "C2", "C3", "C4", "C5", "C6", "C7"]].values


c = (targets == old_targets).mean(1)

train[["C1", "C2", "C3", "C4", "C5", "C6", "C7"]] = targets


train.to_csv(ROOT + "meta_wirbel_dcm_v1.csv", index=False)


# fix folds
folds = (
    pd.read_csv(ROOT + "train_folded_v1.csv", usecols=["StudyInstanceUID", "fold"])
    .set_index("StudyInstanceUID")
    .to_dict()["fold"]
)


train_old = train.copy()


train["fold"] = train["StudyInstanceUID"].map(folds)


train.to_csv(ROOT + "meta_wirbel_dcm_v2.csv", index=False)


dicom_imgs = []
for item in seg_labels:
    study_id = item[:-4]
    fns = os.listdir(f"{ROOT}train_images/{study_id}/")
    dicom_imgs += [f"{study_id}/{fn}" for fn in fns]


set(dicom_imgs) == set(train["Image"].values)


df = pd.read_csv(ROOT + "train.csv")


study_ids = df["StudyInstanceUID"].unique()


def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images.
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img = pydicom.dcmread(path)
    img.PhotometricInterpretation = "YBR_FULL"
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img


def get_dicom_meta(path):
    """
    This supports loading both regular and compressed JPEG images.
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img = pydicom.dcmread(path)
    img.PhotometricInterpretation = "YBR_FULL"
    ipp = img.ImagePositionPatient
    th = img.SliceThickness
    shape = img.pixel_array.shape

    return ipp, th, shape


def do_one(study_id):
    fns = os.listdir(f"{ROOT}train_images/{study_id}/")
    slices = np.sort([int(item[:-4]) for item in fns])
    metas = [get_dicom_meta(f"{ROOT}train_images/{study_id}/{s}.dcm") for s in slices]
    ImagePositionPatient = [m[0] for m in metas]
    SliceThickness = [m[1] for m in metas]
    StudyInstanceUID = [study_id] * len(metas)
    Slice = [s for s in slices]
    Shape = [m[2] for m in metas]
    df_meta = pd.DataFrame({"StudyInstanceUID": StudyInstanceUID, "Slice": Slice})
    df_meta[
        ["ImagePositionPatient_x", "ImagePositionPatient_y", "ImagePositionPatient_z"]
    ] = ImagePositionPatient
    df_meta[["ImageHeight", "ImageWidth"]] = Shape
    df_meta["SliceThickness"] = SliceThickness
    return df_meta


with mp.Pool(16) as p:
    res = list(tqdm(p.imap(do_one, study_ids), total=len(study_ids)))

df_meta = pd.concat(res)


df_meta[["C1", "C2", "C3", "C4", "C5", "C6", "C7", "fold"]] = 0


df_meta["Image"] = (
    df_meta["StudyInstanceUID"] + "/" + df_meta["Slice"].astype(str) + ".dcm"
)

df_meta = df_meta[train.columns]

df_meta.to_csv(ROOT + "test_meta_wirbel_dcm_v1.csv", index=False)

# fix folds
df_meta["fold"] = df_meta["StudyInstanceUID"].map(folds)

df_meta.to_csv(ROOT + "test_meta_wirbel_dcm_v2.csv", index=False)
