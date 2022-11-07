import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pydicom as dicom
import cv2


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


tr_collate_fn = None
val_collate_fn = None


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.df = df.copy()
        self.mode = mode
        self.labels = self.df[self.cfg.classes].values
        self.fns = self.df["Image"].astype(str).values
        self.aug = aug
        if mode == "test":
            self.data_folder = cfg.test_data_folder
        else:
            self.data_folder = cfg.data_folder

    def __getitem__(self, idx):

        label = self.labels[idx]
        img = self.load_one(idx)

        if self.aug:
            img = self.augment(img)

        img = self.normalize_img(img)

        feature_dict = {
            "input": torch.tensor(img).float(),
            "target": torch.tensor(label),
        }
        return feature_dict

    def __len__(self):
        return len(self.fns)

    def load_one(self, idx):
        path = self.data_folder + self.fns[idx]

        if os.path.exists(path):
            if path.split(".")[-1] == "dcm":
                img = dicom.dcmread(path)
                img.PhotometricInterpretation = "YBR_FULL"
                data = img.pixel_array
                data = data - np.min(data)
                if np.max(data) != 0:
                    data = data / np.max(data)
                data = (data * 255).astype(np.uint8)
                img = cv2.resize(data, (self.cfg.image_width, self.cfg.image_height))
                img = img[:, :, np.newaxis]
            else:
                try:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (self.cfg.image_width, self.cfg.image_height))
                    if len(img.shape) > 2:
                        img = img[:, :, 0]
                        img = img[:, :, np.newaxis]
                    else:
                        img = img[:, :, np.newaxis]

                except Exception:
                    print(f"Could not read image {path.split('/')[-1]}.")
        else:
            img = np.zeros((self.cfg.image_width, self.cfg.image_height, 1))
            print(f"Could not read image {path}.")

        assert img.shape == (
            self.cfg.image_width,
            self.cfg.image_height,
            1,
        ), f"shape is {img.shape}, condition failed at {path}"

        return img.transpose(2, 0, 1)

    def augment(self, img):
        img = img.astype(np.float32)
        img = self.aug(image=img.transpose(1, 2, 0))["image"].transpose(2, 0, 1)
        return img

    def normalize_img(self, img):

        if self.cfg.normalization == "image":
            img = (img - img.mean()) / (img.std() + 1e-4)
            img = img.clip(-20, 20)

        elif self.cfg.normalization == "simple":
            img = img / 255

        elif self.cfg.normalization == "min_max":
            img = img - np.min(img)
            img = img / np.max(img)

        return img
