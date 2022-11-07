import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pydicom as dicom
from joblib import Parallel, delayed
from sklearn.utils.class_weight import compute_sample_weight


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


tr_collate_fn = None
val_collate_fn = None


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.df = df.copy()

        self.df = self.df.query('StudyInstanceUID != "1.2.826.0.1.3680043.20574"')

        if cfg.sample_weights:
            self.df["weights"] = compute_sample_weight(
                class_weight="balanced", y=self.df.StudyInstanceUID
            )
            self.weights = self.df["weights"].values
        else:
            self.weights = [1] * len(self.df)

        self.mode = mode
        self.labels = self.df[self.cfg.classes].values
        self.aux_labels = self.df[self.cfg.aux_classes].values
        self.fns = self.df["Image"].astype(str).values

        self.cache = {}

        self.aug = aug
        if mode == "test":
            self.data_folder = cfg.test_data_folder
        else:
            self.data_folder = cfg.data_folder

    def __getitem__(self, idx):

        label = self.labels[idx]
        aux_label = self.aux_labels[idx]
        img, frame_ids = self.load_one(idx)
        frame_ids = np.array([int(x) for x in frame_ids])

        if self.aug:
            img = self.augment(img)

        img = self.normalize_img(img)

        if self.mode == "train" and np.random.random() <= self.cfg.zflip_probability:
            img = np.ascontiguousarray(img[::-1])
            frame_ids = np.ascontiguousarray(frame_ids[::-1])

        feature_dict = {
            "input": torch.tensor(img).float(),
            "target": torch.tensor(label),
            "vert_target": torch.tensor(aux_label),
            "weight": torch.tensor(self.weights[idx]),
            "frame_id": torch.tensor(frame_ids),
        }
        return feature_dict

    def __len__(self):
        return len(self.fns)

    def load_frame(self, img_name, frame_num, fr):
        img_path = img_name.replace(f"/{frame_num}.", f"/{fr}.")
        path = os.path.join(self.data_folder, img_path)
        img = self.read_image(path)

        if self.cfg.stack_neighbor_frames > 0:
            imgs = []

            for j in list(range(1, self.cfg.stack_neighbor_frames + 1))[::-1]:
                p = path.replace(f"/{fr}.", f"/{fr-j}.")
                imgs.append(self.read_image(p))

            imgs.append(img)

            for j in list(range(1, self.cfg.stack_neighbor_frames + 1)):
                p = path.replace(f"/{fr}.", f"/{fr+j}.")
                imgs.append(self.read_image(p))

            img = np.concatenate(imgs, axis=0)

            return img.astype(np.uint8)

    def load_one(self, idx):
        img_name = self.fns[idx]
        frame_num = int(img_name.split("/")[-1].split(".")[0])

        img_stack = []
        # 2.5D CNN
        frame_nums = [frame_num]
        curr_step = self.cfg.frames_step_size
        for _ in range(self.cfg.frames_num):
            frame_nums.insert(0, frame_num - curr_step)
            frame_nums.append(frame_num + curr_step)
            curr_step += self.cfg.frames_step_size

        img_stack = Parallel(
            n_jobs=self.cfg.num_workers_dataloader, backend="multiprocessing"
        )(delayed(self.load_frame)(img_name, frame_num, fr) for fr in frame_nums)
        img_stack = np.concatenate(img_stack, axis=0)

        return img_stack.astype(np.uint8), frame_nums

    def augment(self, img):
        img = self.aug(image=img.transpose(1, 2, 0))["image"].transpose(2, 0, 1)
        return img

    def normalize_img(self, img):
        img = img.astype(np.float32)

        if self.cfg.normalization == "image":
            img = (img - img.mean()) / (img.std() + 1e-4)
            img = img.clip(-20, 20)

        elif self.cfg.normalization == "simple":
            img = img / 255

        elif self.cfg.normalization == "min_max":
            img = img - np.min(img)
            img = img / np.max(img)

        return img

    def manual_convert(self, arr, dset):
        return (arr * dset.RescaleSlope + dset.RescaleIntercept).clip(
            self.cfg.min_dicom_level, self.cfg.max_dicom_level
        )

    def read_image(self, path):
        if (
            self.mode != "train"
            and self.cfg.cache_disk != ""
            and os.path.exists(f"{self.cfg.cache_disk}/{path.replace('/','_')}.npy")
        ):
            img = np.load(f"{self.cfg.cache_disk}/{path.replace('/','_')}.npy")
        elif self.mode != "train" and path in self.cache:
            img = self.cache[path]
        elif os.path.exists(path):
            img = dicom.dcmread(path)
            img.PhotometricInterpretation = self.cfg.dicom_interpretation
            if self.cfg.manual_convert:
                data = self.manual_convert(img.pixel_array, img)
            else:
                data = img.pixel_array
            data = data - np.min(data)
            if np.max(data) != 0:
                data = data / np.max(data)
            data = (data * 255).astype(np.uint8)
            img = data[:, :, np.newaxis]
            img = self.cfg.resize_aug(image=img)["image"]

            if self.mode != "train":

                if self.cfg.cache_disk != "":
                    np.save(f"{self.cfg.cache_disk}/{path.replace('/','_')}", img)
                else:
                    self.cache[path] = img

                    if len(self.cache) >= self.cfg.cache_size:
                        self.cache.pop(next(iter(self.cache)))

        else:
            img = np.zeros(
                (self.cfg.image_width_loading, self.cfg.image_height_loading, 1),
                dtype=np.uint8,
            )
            # print(f"Could not read image {path}.")

        assert img.shape == (
            self.cfg.image_width_loading,
            self.cfg.image_height_loading,
            1,
        ), f"shape is {img.shape}, condition failed at {path}"

        return img.transpose(2, 0, 1)
