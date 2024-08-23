from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from model.layers import LinearBnRelu
from model.backbones import BackboneGetter
from torch import Tensor

optimizers: dict[str, type[optim.Optimizer]] = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop
}


class FaceFeaturesExtractor(pl.LightningModule):

    def __init__(
            self,
            backbone_getter: BackboneGetter,
            learning_rate: float = 1e-3,
            optimizer: str = "adam",
            output_layer_depth: int = 4,
    ) -> None:
        super().__init__()
        self.backbone, classifier_in_features = backbone_getter()

        output_layers: list[nn.Module] = []
        for i in range(output_layer_depth):
            out_features = classifier_in_features // 2
            output_layers.append(LinearBnRelu(classifier_in_features, out_features))
            classifier_in_features = out_features
        output_layers.append(nn.Linear(classifier_in_features, 8))

        self.out = nn.Sequential(*output_layers)
        self.optimizer = optimizers[optimizer](params=self.parameters(), lr=learning_rate)

    def forward(self, images: torch.tensor):
        features = self.backbone(images)
        return self.out(features)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, Tensor | Any]:
        avg_loss, [age_loss, gender_loss, race_loss] = self._common_step(batch)

        self.log_dict({
            "loss": avg_loss,
            "age_loss": age_loss,
            "gender_loss": gender_loss,
            "race_loss": race_loss
        }, on_step=False, on_epoch=True)
        return {"loss": avg_loss, "age_loss": age_loss, "gender_loss": gender_loss, "race_loss": race_loss}

    def _common_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        images, targets = batch
        predictions = self.forward(images)

        age, gender, race = targets[0], targets[1:3], targets[3:]
        predicted_age, predicted_gender, predicted_race = predictions[0], predictions[1:3], predictions[3:]

        age_loss = F.mse_loss(predicted_age, age)
        gender_loss = F.cross_entropy(predicted_gender, gender)
        race_loss = F.cross_entropy(predicted_race, race)
        avg_loss = (age_loss + gender_loss + race_loss) / 3

        return avg_loss, [age_loss, gender_loss, race_loss]

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, Tensor | Any]:
        avg_loss, [age_loss, gender_loss, race_loss] = self._common_step(batch)
        self.log_dict({
            "loss": avg_loss,
            "age_loss": age_loss,
            "gender_loss": gender_loss,
            "race_loss": race_loss
        }, on_step=False, on_epoch=True)
        return {"loss": avg_loss, "age_loss": age_loss, "gender_loss": gender_loss, "race_loss": race_loss}

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        avg_loss, _ = self._common_step(batch)
        return avg_loss

    def predict_step(self, images: torch.tensor) -> tuple[int, int, int]:
        output = self.forward(images)
        age = int(output[0])
        gender = torch.argmax(F.sigmoid(output[1:3])).item()
        race = torch.argmax(F.softmax(output[3:])).item()
        return age, gender, race

    def configure_optimizers(self) -> optim.Optimizer:
        return self.optimizer
