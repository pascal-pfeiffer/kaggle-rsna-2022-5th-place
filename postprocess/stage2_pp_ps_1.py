import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


def post_process_pipeline(cfg, val_data, val_df):

    #print(val_data)
    #print(val_df)

    # Category: Weight
    # Vertebrae negative: 1
    # Vertebrae positive: 2
    # Patient negative: 7
    # Patient positive: 14
    # https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/340392

    # means = [train["C1"].mean(),
    # train["C2"].mean(),
    # train["C3"].mean(),
    # train["C4"].mean(),
    # train["C5"].mean(),
    # train["C6"].mean(),
    # train["C7"].mean(),
    # train["patient_overall"].mean()]

    means = [
        0.07231302625061913,
        0.1411589895988113,
        0.03615651312530956,
        0.05349182763744428,
        0.08023774145616643,
        0.13719663199603765,
        0.19465081723625557,
        0.4759782070331847,
    ]

    preds = val_data["logits"].detach().sigmoid().cpu().numpy()

    val_df[
        [
            "pred_frac_c1",
            "pred_frac_c2",
            "pred_frac_c3",
            "pred_frac_c4",
            "pred_frac_c5",
            "pred_frac_c6",
            "pred_frac_c7",
        ]
    ] = preds

    window_size = 7

    uids = val_df['StudyInstanceUID'].unique()

    for c in  [
        "pred_frac_c1",
        "pred_frac_c2",
        "pred_frac_c3",
        "pred_frac_c4",
        "pred_frac_c5",
        "pred_frac_c6",
        "pred_frac_c7",
    ]:
        val_df[c] = val_df.groupby("StudyInstanceUID")[c].rolling(window_size).mean().reset_index(0,drop=True)

    preds = (
        val_df.groupby("StudyInstanceUID")[
            [
                "pred_frac_c1",
                "pred_frac_c2",
                "pred_frac_c3",
                "pred_frac_c4",
                "pred_frac_c5",
                "pred_frac_c6",
                "pred_frac_c7",
            ]
        ]
        .max()
        .reset_index(drop=True)
    )
    preds["pred_overall"] = 1 - (
        (1 - preds["pred_frac_c1"])
        * (1 - preds["pred_frac_c2"])
        * (1 - preds["pred_frac_c3"])
        * (1 - preds["pred_frac_c4"])
        * (1 - preds["pred_frac_c5"])
        * (1 - preds["pred_frac_c6"])
        * (1 - preds["pred_frac_c7"])
    )

    for vert in range(8):
        preds.values[:, vert] = np.clip((preds.values[:, vert] / preds.values[:, vert].mean())* 2* means[vert]/ (1 + means[vert]),
            0.01,
            0.99,
        )

    return preds
