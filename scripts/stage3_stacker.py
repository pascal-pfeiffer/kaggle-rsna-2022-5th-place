import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from torch import nn
from transformers import get_cosine_schedule_with_warmup
import os
from torch.utils.data import Dataset, DataLoader


ROOT = "./input/rsna-2022-cervical-spine-fracture-detection/"
OUT = "./output/models/"


df = pd.read_csv(ROOT + "labels_fracture_2D_v1_fold-1.csv")
df = df.query('StudyInstanceUID != "1.2.826.0.1.3680043.20574"').reset_index(drop=True)
train_df = df


exp_names = [
    "stage2_cfg_ps_wd_29_val_ff",
    "stage2_cfg_ps_wd_30_val_ff",
    "stage2_cfg_ps_wd_37_val_ff",
    "stage2_cfg_ps_wd_40_val_ff",
]
seeds = ["42", "42", "42", "42"]


train_datas = []
for j in range(len(exp_names)):
    exp_name = exp_names[j]

    seed = seeds[j]

    val_data = torch.load(
        f"{OUT}{exp_name}/fold-1/test_data_seed{seed}.pth",
        map_location="cpu",
    )
    train_datas.append(val_data["logits"].detach().float().sigmoid().cpu().numpy())


preds_train = np.mean(train_datas, axis=0)


pred_cols = [
    "pred_frac_c1",
    "pred_frac_c2",
    "pred_frac_c3",
    "pred_frac_c4",
    "pred_frac_c5",
    "pred_frac_c6",
    "pred_frac_c7",
]

label_cols = [
    "label_frac_c1",
    "label_frac_c2",
    "label_frac_c3",
    "label_frac_c4",
    "label_frac_c5",
    "label_frac_c6",
    "label_frac_c7",
]


train_df[pred_cols] = preds_train


class RSNAStackerDataset(Dataset):
    def __init__(self, df, mode):
        self.df = df.copy().reset_index(drop=True)
        self.mode = mode

        self.feature_cols = []
        self.label_cols = label_cols.copy()

        df = self.df

        features = []

        for j, l in enumerate(pred_cols):
            features.append(
                df.groupby("StudyInstanceUID")[l].mean().values.reshape(-1, 1)
            )
            features.append(
                df.groupby("StudyInstanceUID")[l].min().values.reshape(-1, 1)
            )
            features.append(
                df.groupby("StudyInstanceUID")[l].max().values.reshape(-1, 1)
            )

        features.append(
            df.groupby("StudyInstanceUID").size().values.reshape(-1, 1) / 1_000
        )

        labels = (
            df.groupby("StudyInstanceUID")[self.label_cols].max().reset_index(drop=True)
        )
        labels["label_overall"] = labels[self.label_cols].max(axis=1)

        self.X = np.concatenate(features, axis=1)
        self.y = labels.values

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        return torch.FloatTensor(X), torch.FloatTensor(y)

    def __len__(self):
        return self.df.StudyInstanceUID.nunique()


ds = RSNAStackerDataset(train_df.copy(), mode="train")


def weighted_loss(y_pred_logit, y, verbose=False):
    """
    Weighted loss
    We reuse torch.nn.functional.binary_cross_entropy_with_logits here. pos_weight and
    weights combined give us necessary coefficients described in
    https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/340392

    See also this explanation:
    https://www.kaggle.com/code/samuelcortinhas/rsna-fracture-detection-in-depth-eda/notebook
    """

    competition_weights = {
        "-": torch.tensor([1, 1, 1, 1, 1, 1, 1, 7], dtype=torch.float),
        "+": torch.tensor([2, 2, 2, 2, 2, 2, 2, 14], dtype=torch.float),
    }

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_pred_logit,
        y,
        reduction="none",
    )

    if verbose:
        print("loss", loss)

    weights = y * competition_weights["+"] + (1 - y) * competition_weights["-"]

    loss = loss * weights

    loss = loss.sum()
    loss = loss / weights.sum()

    return loss


class RSNAStackerModel(nn.Module):
    def __init__(self, n_features):
        super(RSNAStackerModel, self).__init__()

        self.sizes = [256, 128, 64]

        self.features = nn.Sequential(
            nn.Linear(n_features, self.sizes[0]),
            nn.PReLU(),
            nn.Linear(self.sizes[0], self.sizes[1]),
            nn.PReLU(),
            nn.Linear(self.sizes[1], self.sizes[2]),
            nn.PReLU(),
            nn.BatchNorm1d(self.sizes[-1]),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(self.sizes[-1], 8)

        self.loss_fn = weighted_loss

    def forward(self, x, y):

        x = self.features(x)
        x = self.head(x)

        output = {}

        output["logits"] = x

        if self.training:
            output["loss"] = self.loss_fn(x, y)

        return output


LR = 0.001
BATCH_SIZE = 32
EPOCHS = 20

exp_name = "final_nn_v0_ff"

if not os.path.exists(f"nn_models/{exp_name}"):
    os.makedirs(f"nn_models/{exp_name}")


seed_preds = []
for seed in range(5):

    DEVICE = "cpu"

    train_ds = RSNAStackerDataset(train_df.copy(), mode="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    model = RSNAStackerModel(n_features=len(train_ds[0][0]))
    model.to(DEVICE)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=EPOCHS * len(train_loader)
    )

    model.train()

    for e in tqdm(range(EPOCHS)):
        tbar = tqdm(train_loader, disable=True)

        loss_list = []

        model.train()

        for idx, data in enumerate(tbar):
            data = [x.to(DEVICE) for x in data]
            inputs, target = data

            optimizer.zero_grad()
            output = model(inputs, target)

            loss = output["loss"]

            loss.backward()
            optimizer.step()

            loss_list.append(loss.detach().cpu().item())

            avg_loss = np.round(np.mean(loss_list), 4)

            tbar.set_description(
                f"Ep {e+1} Loss: {avg_loss} lr: {optimizer.param_groups[0]['lr']:.5f}"
            )

            scheduler.step()

    torch.save(model.state_dict(), f"nn_models/{exp_name}/checkpoint_seed{seed}.pth")
