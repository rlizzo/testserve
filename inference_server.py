from __future__ import annotations

import torch
import torchvision

from gridserve_sdk import Composition, GridModel, ModelComponent, expose
from gridserve_sdk.types import Image, Label


class ClassificationInference(ModelComponent):
    def __init__(self, model):  # skipcq: PYL-W1401, PYL-W0621
        self.model = model

    @expose(
        inputs={"img": Image()},
        outputs={"prediction": Label(path="imagenet_labels.txt")},
    )
    def classify(self, img):
        img = img.float() / 255
        mean = torch.tensor([[[0.485, 0.456, 0.406]]]).float()
        std = torch.tensor([[[0.229, 0.224, 0.225]]]).float()
        img = (img - mean) / std
        img = img.permute(0, 3, 2, 1)
        out = self.model(img)
        return out.argmax()


model = torchvision.models.resnet18(pretrained=True).eval()
torch.jit.script(model).save("resnet.pt")
resnet = GridModel("resnet.pt")
comp = ClassificationInference(resnet)
composition = Composition(classification=comp)
composition.serve()
