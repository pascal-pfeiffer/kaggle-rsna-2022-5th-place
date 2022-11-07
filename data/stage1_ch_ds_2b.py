import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
            self.mask_folder = cfg.mask_folder

    def __getitem__(self, idx):

        label = self.labels[idx]
        img = self.load_one(idx)
        if self.mode == "test":
            if self.aug:
                img, _ = self.augment(img, mask=None)

            img = self.normalize_img(img)
            torch_img = torch.tensor(img).float().permute(2, 0, 1)

            feature_dict = {
                "input": torch_img,
                "target": torch.tensor(label),
            }
            return feature_dict

        else:
            try:
                mask = self.load_mask(idx)
            except:
                mask = np.zeros((img.shape[0], img.shape[1]))

        if self.aug:
            img, mask = self.augment(img, mask)

        img = self.normalize_img(img)
        torch_img = torch.tensor(img).float().permute(2, 0, 1)

        feature_dict = {
            "input": torch_img,
            "mask": torch.tensor(mask),
            "target": torch.tensor(label),
        }
        return feature_dict

    def __len__(self):
        return len(self.fns)

    def load_mask(self, idx):

        path = self.mask_folder + self.fns[idx].replace("dcm", "png")
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        mask[mask > 7] = 0
        return mask

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
                img = data[:, :, np.newaxis]
            else:
                try:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
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

        return img

    def augment(self, img, mask):
        img = img.astype(np.float32)
        if mask is not None:
            transformed = self.aug(image=img, mask=mask)
            trans_img = transformed["image"]
            trans_mask = transformed["mask"]

            return trans_img, trans_mask
        else:
            transformed = self.aug(image=img)
            trans_img = transformed["image"]

            return trans_img, None

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
