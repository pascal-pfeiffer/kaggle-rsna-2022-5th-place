from torchvision.ops import masks_to_boxes
import torch
import importlib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import glob
from tqdm import tqdm
import sys
import pydicom


sys.path.append("./configs")
sys.path.append("./data")
sys.path.append("./models")
sys.path.append("./postprocess")

ROOT = "./input/rsna-2022-cervical-spine-fracture-detection/"

df = pd.read_csv(ROOT + "test_meta_wirbel_dcm_v2.csv")

df["frame"] = df["Slice"]
df = df.sort_values(["StudyInstanceUID", "frame"]).reset_index(drop=True)

df[
    [
        "label_frac_c1",
        "label_frac_c2",
        "label_frac_c3",
        "label_frac_c4",
        "label_frac_c5",
        "label_frac_c6",
        "label_frac_c7",
    ]
] = 0
df[
    [
        "label_vert_c1",
        "label_vert_c2",
        "label_vert_c3",
        "label_vert_c4",
        "label_vert_c5",
        "label_vert_c6",
        "label_vert_c7",
    ]
] = 0


def load_seg_model(cfg_name):
    cfg = importlib.import_module(cfg_name)
    importlib.reload(cfg)
    cfg = cfg.cfg
    # print(cfg)
    print(cfg.model, cfg.dataset, cfg.backbone, cfg.image_width, cfg.image_height)

    cfg.mixed_precision = False

    ds = importlib.import_module(cfg.dataset)
    importlib.reload(ds)
    CustomDataset = ds.CustomDataset
    # collate_fn = importlib.import_module(cfg.dataset).collate_fn
    batch_to_device = ds.batch_to_device

    cfg.post_process_pipeline = importlib.import_module(
        cfg.post_process_pipeline
    ).post_process_pipeline

    m = importlib.import_module(cfg.model)
    importlib.reload(m)
    Net = m.Net

    # test settings
    cfg.data_folder = f"{ROOT}train_images/"
    cfg.test_data_folder = f"{ROOT}train_images/"
    cfg.data_dir = ROOT
    cfg.pretrained = False
    cfg.device = "cuda"
    cfg.return_logits = True

    cfg.calc_loss = False

    state_dicts = []
    for filepath in glob.iglob(f"./output/models/{cfg_name}/fold-1/checkpoint_*.pth"):

        state_dicts.append(filepath)
        break
    print(state_dicts)

    nets = []
    for i in range(len(state_dicts)):
        d = torch.load(state_dicts[i])["model"]
        new_d = {}
        for k, v in d.items():
            new_d[k.replace("module.", "")] = v
        sd = new_d

        net = Net(cfg).eval().to(cfg.device)
        net.load_state_dict(sd)

        nets.append(net)

    print("-------------")
    return nets, cfg, CustomDataset, batch_to_device


net, cfg, CustomDataset, batch_to_device = load_seg_model("stage1_S1B")


df[["C1", "C2", "C3", "C4", "C5", "C6", "C7"]] = 0


cfg.batch_size = 64
cfg.cache_size = 200
cfg.cache_disk = ""


boxes = []
with torch.inference_mode():

    test_ds = CustomDataset(df, cfg, cfg.val_aug, mode="test")
    test_dl = DataLoader(
        test_ds, shuffle=False, batch_size=cfg.batch_size, num_workers=2
    )

    for batch in tqdm(test_dl):
        batch = batch_to_device(batch, "cuda")
        out = net[0](batch)
        pred = (out["logits"].sigmoid().max(1)[0] > 0.5).long()
        box = torch.zeros((pred.shape[0], 4))

        not_empty = pred.sum((1, 2)) > 10
        b = masks_to_boxes(pred[not_empty])
        box[not_empty] = b.cpu()
        boxes += [box]

boxes = torch.cat(boxes)

box_preds = df[["StudyInstanceUID"]].copy()
box_preds[["x1", "y1", "x2", "y2"]] = boxes.numpy()


# need original img shape to scale boxes to fit original image shape
def get_dicom_meta(path):
    """
    This supports loading both regular and compressed JPEG images.
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img = pydicom.dcmread(path)
    img.PhotometricInterpretation = "YBR_FULL"
    shape = img.pixel_array.shape

    return shape


meta = df[["Image", "StudyInstanceUID"]].drop_duplicates("StudyInstanceUID")
img_fns = [f"{ROOT}train_images/{fn}" for fn in meta["Image"].values]
shapes = [get_dicom_meta(p) for p in img_fns]
meta[["ImageHeight", "ImageWidth"]] = shapes

scales = (
    meta[["StudyInstanceUID", "ImageHeight", "ImageWidth"]]
    .drop_duplicates()
    .set_index("StudyInstanceUID")
    .to_dict()
)


image_width = 320
image_height = 320

image_width_orig = 360
image_height_orig = 360

pad_ = (image_height_orig - image_height) // 2


def get_box(study_id):
    df = box_preds[box_preds["StudyInstanceUID"] == study_id].copy()
    raw_boxes = df[["x1", "y1", "x2", "y2"]].values
    df["area"] = (raw_boxes[:, 2] - raw_boxes[:, 0]) * (
        raw_boxes[:, 3] - raw_boxes[:, 1]
    )
    raw_boxes2 = raw_boxes[
        ((raw_boxes[:, 2] - raw_boxes[:, 0]) * (raw_boxes[:, 3] - raw_boxes[:, 1])) > 9
    ]
    raw_boxes2 = raw_boxes2 + pad_  # account for center crop

    x_scale = scales["ImageWidth"][study_id] / image_width_orig
    y_scale = scales["ImageHeight"][study_id] / image_height_orig

    try:
        x1y1 = np.quantile(raw_boxes2[:, :2], 0.05, axis=0)
        x2y2 = np.quantile(raw_boxes2[:, 2:], 0.95, axis=0)
        z1 = np.quantile(np.where(df["area"].values > 0)[0], 0.05)
        z2 = np.quantile(np.where(df["area"].values > 0)[0], 0.95)

        box = np.array(
            [
                x_scale * x1y1[0],
                x_scale * x2y2[0],
                y_scale * x1y1[1],
                y_scale * x2y2[1],
                z1,
                z2,
            ]
        )
    except:
        print(study_id)
        box = np.array([0, 0, 0, 0, 0, 0])
    return box


boxes = [get_box(study_id) for study_id in tqdm(meta["StudyInstanceUID"].values)]
meta[["x1", "x2", "y1", "y2", "z1", "z2"]] = boxes
meta.to_csv(ROOT + "crop_box_3d.csv", index=False)

print(meta)
