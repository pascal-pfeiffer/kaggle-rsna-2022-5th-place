import pandas as pd
import numpy as np
import torch

ROOT = "./input/rsna-2022-cervical-spine-fracture-detection/"

df_meta = pd.read_csv(ROOT + "test_meta_wirbel_dcm_v2.csv")

preds = torch.load("./output/models/stage1_S1A/fold-1/test_data_seed42.pth")

preds = preds["logits"].sigmoid().cpu().numpy()

df_meta[[f"pred_C{i}" for i in range(1, 8)]] = preds

train = pd.read_csv(ROOT + "train_folded_v1.csv")

df_meta["label_vert_c1"] = (df_meta["pred_C1"] > 0.5).astype(int)
df_meta["label_vert_c2"] = (df_meta["pred_C2"] > 0.5).astype(int)
df_meta["label_vert_c3"] = (df_meta["pred_C3"] > 0.5).astype(int)
df_meta["label_vert_c4"] = (df_meta["pred_C4"] > 0.5).astype(int)
df_meta["label_vert_c5"] = (df_meta["pred_C5"] > 0.5).astype(int)
df_meta["label_vert_c6"] = (df_meta["pred_C6"] > 0.5).astype(int)
df_meta["label_vert_c7"] = (df_meta["pred_C7"] > 0.5).astype(int)

df_meta = df_meta.merge(train, on="StudyInstanceUID")

df_meta["label_frac_c1"] = df_meta["label_vert_c1"] * df_meta["C1_y"]
df_meta["label_frac_c2"] = df_meta["label_vert_c2"] * df_meta["C2_y"]
df_meta["label_frac_c3"] = df_meta["label_vert_c3"] * df_meta["C3_y"]
df_meta["label_frac_c4"] = df_meta["label_vert_c4"] * df_meta["C4_y"]
df_meta["label_frac_c5"] = df_meta["label_vert_c5"] * df_meta["C5_y"]
df_meta["label_frac_c6"] = df_meta["label_vert_c6"] * df_meta["C6_y"]
df_meta["label_frac_c7"] = df_meta["label_vert_c7"] * df_meta["C7_y"]
df_meta["label_overall"] = df_meta[
    ["C1_y", "C2_y", "C3_y", "C4_y", "C5_y", "C6_y", "C7_y"]
].max(axis=1)

df_meta["fold"] = df_meta["fold_x"]

df_meta = df_meta[
    [c for c in df_meta.columns if not c in ["image_name", "UID", "frame_id"]]
]


df_meta.to_csv(ROOT + "labels_fracture_2D_v1_ff.csv", index=False)
