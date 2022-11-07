import os
import numpy as np
import pandas as pd
import pydicom


ROOT = "./input/rsna-2022-cervical-spine-fracture-detection/"

train = pd.read_csv(ROOT + "train.csv")


# https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/346981


def get_z_direction(dicom_1_path, dicom_last_path):

    dicom_1 = pydicom.dcmread(dicom_1_path)
    dicom_last = pydicom.dcmread(dicom_last_path)

    if dicom_1.ImagePositionPatient[2] - dicom_last.ImagePositionPatient[2] > 0:
        return True
    else:
        return False


res = []
for item in train["StudyInstanceUID"].unique():
    dicoms = os.listdir(f"{ROOT}train_images/{item}/")
    dicoms = np.sort([int(item[:-4]) for item in dicoms])
    dicom_1_path = f"{ROOT}train_images/{item}/{dicoms.min()}.dcm"
    dicom_last_path = f"{ROOT}train_images/{item}/{dicoms.max()}.dcm"
    d = get_z_direction(dicom_1_path, dicom_last_path)
    res += [d]

res = np.array(res)


reversed_ids = np.where(res != True)[0]


reversed_dicoms = train["StudyInstanceUID"].unique()[reversed_ids]


pd.Series(reversed_dicoms).to_csv(
    ROOT + "reversed_dicoms.txt", index=False, header=None
)
