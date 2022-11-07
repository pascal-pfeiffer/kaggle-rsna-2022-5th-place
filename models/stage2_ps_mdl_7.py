from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import timm
from timm.models.efficientnet_blocks import InvertedResidual
from torch.distributions import Beta
import numpy as np


def create_timm_model(model_name, pretrained, in_chans):

    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
        drop_rate=0,
        in_chans=in_chans,
    )

    return model


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y, Y2, sample_weights):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        elif n_dims == 4:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]
        else:
            X = (
                coeffs.view(-1, 1, 1, 1, 1) * X
                + (1 - coeffs.view(-1, 1, 1, 1, 1)) * X[perm]
            )

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
            Y2 = (Y2 + Y2[perm]).clip(0, 1)
        else:
            if len(Y.shape) == 1:
                Y = coeffs * Y + (1 - coeffs) * Y[perm]
                Y2 = coeffs * Y2 + (1 - coeffs) * Y2[perm]
            else:
                Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]
                Y2 = coeffs.view(-1, 1) * Y2 + (1 - coeffs.view(-1, 1)) * Y2[perm]

        sample_weights = (
            coeffs.view(-1, 1) * sample_weights
            + (1 - coeffs.view(-1, 1)) * sample_weights[perm]
        )

        return X, Y, Y2, sample_weights


class SequenceLayer(nn.Module):
    def __init__(
        self, seq_length, in_channels, out_channels, kernel_size, padding, bias
    ):
        super(SequenceLayer, self).__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, kernel_size[0], kernel_size[1]),
            padding=(1, padding[0], padding[1]),
            bias=bias,
        )
        self.seq_length = seq_length

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.view(bs // self.seq_length, self.seq_length, c, h, w)
        x = self.conv(x.transpose(1, 2).contiguous()).transpose(2, 1).contiguous()
        x = x.flatten(0, 1)
        return x


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
        "-": torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.float, device=y.device),
        "+": torch.tensor([2, 2, 2, 2, 2, 2, 2], dtype=torch.float, device=y.device),
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


class Net(nn.Module):
    def __init__(self, cfg: Any):
        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = 7
        self.n_aux_classes = 7
        self.backbone = self._create_backbone()

        self.backbone.conv_stem.stride = self.cfg.stride
        print("Conv stem", self.backbone.conv_stem)

        if self.cfg.loss == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            self.loss_fn_aux = nn.BCEWithLogitsLoss(reduction="none")
        elif self.cfg.loss == "weighted_bce":
            self.loss_fn = weighted_loss
            self.loss_fn_aux = nn.BCEWithLogitsLoss(reduction="none")

        self.mixup = Mixup(mix_beta=cfg.mix_beta, mixadd=cfg.mixadd)

        if cfg.pool == "gem":
            self.pooling = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.pooling = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif cfg.pool == "max":
            self.pooling = nn.AdaptiveMaxPool2d(1)

        self.head = torch.nn.Linear(self.backbone.num_features, self.n_classes)
        self.aux_head = torch.nn.Linear(self.backbone.num_features, self.n_aux_classes)

        self.ids_prev = []
        self.blocks_prev = None

    def _create_backbone(self) -> torch.nn.Module:
        backbone = create_timm_model(
            model_name=self.cfg.backbone,
            pretrained=self.cfg.pretrained,
            in_chans=self.cfg.image_channels,
        )

        self.seq_length = self.cfg.frames_num * 2 + 1
        print("SEQ LENGTH", self.seq_length)

        orig_conv = backbone.conv_head
        sequence_conv = SequenceLayer(
            self.seq_length,
            orig_conv.in_channels,
            orig_conv.out_channels,
            orig_conv.kernel_size,
            orig_conv.padding,
            orig_conv.bias,
        )
        for j in range(3):
            sequence_conv.conv.weight.data[:, :, j, :, :].copy_(
                orig_conv.weight.data / 3
            )
        backbone.conv_head = sequence_conv

        num_layers = self.cfg.num_3d_layers

        for module in list(backbone.modules())[::-1]:
            if num_layers == 0:
                break

            if isinstance(module, InvertedResidual):

                orig_conv = module.conv_pw
                sequence_conv = SequenceLayer(
                    self.seq_length,
                    orig_conv.in_channels,
                    orig_conv.out_channels,
                    orig_conv.kernel_size,
                    orig_conv.padding,
                    orig_conv.bias,
                )
                for j in range(3):
                    sequence_conv.conv.weight.data[:, :, j, :, :].copy_(
                        orig_conv.weight.data / 3
                    )
                module.conv_pw = sequence_conv
                num_layers -= 1

        return backbone

    def forward_backbone_2d(self, xxx):
        xxx = self.backbone.conv_stem(xxx)
        xxx = self.backbone.bn1(xxx)
        xxx = self.backbone.blocks[:-1](xxx)
        xxx = self.backbone.blocks[-1][: -self.cfg.num_3d_layers](xxx)
        return xxx

    def forward_backbone(self, x, ids):

        ids = ids.cpu().numpy()

        idx = []
        id_map = {}
        idx_map = {}
        idx_map_prev = {}
        for jj, id in enumerate(ids):
            if id in self.ids_prev:
                idx_map_prev[jj] = self.ids_prev.index(id)
            elif id not in id_map:
                idx.append(jj)
                id_map[id] = jj
            else:
                idx_map[jj] = id_map[id]

        xxx = self.forward_backbone_2d(x[idx])
        xx = torch.zeros((len(ids), xxx.size(1), xxx.size(2), xxx.size(3))).to(
            xxx.device
        )
        if self.cfg.mixed_precision:
            xx = xx.half()
        xx[idx] = xxx

        for k, v in idx_map_prev.items():
            xx[k] = self.blocks_prev[v]

        for k, v in idx_map.items():
            xx[k] = xx[v]

        self.ids_prev = list(ids)
        self.blocks_prev = xx.clone()

        x = self.backbone.blocks[-1][-self.cfg.num_3d_layers :](xx)
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)

        return x

    def forward(
        self,
        batch: Dict,
        calculate_loss: bool = True,
    ) -> Dict:
        x = batch["input"]
        targets = batch["target"].float()
        vert_targets = batch["vert_target"].float()
        sample_weights = batch["weight"].float()

        if np.random.random() <= self.cfg.mixup_probability and self.training:
            x, targets, vert_targets, sample_weights = self.mixup(
                x, targets, vert_targets, sample_weights
            )

        images = []
        for i in range(0, x.size(0), 1):
            img = x[i]
            img_curr = []
            for j in range(0, img.size(0), self.cfg.image_channels):
                img_curr.append(img[j : j + self.cfg.image_channels])
            img_curr = torch.stack(img_curr, dim=0).permute(1, 0, 2, 3)
            images.append(img_curr)

        x = torch.stack(images, dim=0)

        x = x.permute(0, 2, 1, 3, 4)

        x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))

        if self.training:
            x = self.backbone(x)
        else:
            ids = batch["frame_id"]
            ids = ids.reshape(ids.size(0) * ids.size(1))
            x = self.forward_backbone(x, ids)

        x = self.pooling(x)[:, :, 0, 0]

        if self.cfg.drop_out > 0.0:
            x = F.dropout(x, p=self.cfg.drop_out, training=self.training)

        res = []
        for i in range(0, x.size(0), self.seq_length):

            if self.cfg.use_3d_centerpooling:
                res.append(x[i + (self.seq_length - 1) // 2].squeeze(0))
            else:
                res.append(x[i : i + self.seq_length].mean(0).squeeze(0))

        x = torch.stack(res, dim=0)

        logits = self.head(x)
        aux_logits = self.aux_head(x)

        outputs = {}

        if not self.training:
            outputs["logits"] = logits
            outputs["vert_logits"] = aux_logits

        if calculate_loss:
            outputs["target"] = targets
            outputs["vert_target"] = vert_targets

            frac_loss = self.loss_fn(logits, targets).mean(dim=1)
            vert_loss = self.loss_fn_aux(aux_logits, vert_targets).mean(dim=1)

            loss = frac_loss + self.cfg.aux_loss_weight * vert_loss

            if self.training and self.cfg.sample_weights:
                loss = loss * sample_weights

                loss = loss.sum()
                loss = loss / sample_weights.sum()
            else:
                loss = loss.mean()

            outputs["frac_loss"] = frac_loss.mean()
            outputs["vert_loss"] = vert_loss.mean()
            outputs["loss"] = loss

        return outputs
