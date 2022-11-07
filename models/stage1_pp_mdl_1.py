from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn
import timm
from torch.nn.parameter import Parameter


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


class Net(nn.Module):
    def __init__(self, cfg: Any):

        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = 7
        self.backbone = self._create_backbone()
        self.loss_fn = nn.BCEWithLogitsLoss()

        if cfg.pool == "gem":
            self.pooling = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.pooling = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.pooling = nn.AdaptiveAvgPool2d(1)

        self.head = torch.nn.Linear(self.backbone.num_features, self.n_classes)

    def _create_backbone(self) -> torch.nn.Module:
        backbone = create_timm_model(
            model_name=self.cfg.backbone, pretrained=True, in_chans=1
        )

        return backbone

    def forward(
        self,
        batch: Dict,
        calculate_loss: bool = True,
    ) -> Dict:
        x = batch["input"]
        x = self.backbone(x)
        x = self.pooling(x)[:, :, 0, 0]
        logits = self.head(x)

        outputs = {}

        if not self.training:
            outputs["logits"] = logits

        if calculate_loss:
            outputs["target"] = batch["target"].float()
            targets = batch["target"].float()

            outputs["loss"] = self.loss_fn(logits, targets)

        return outputs
