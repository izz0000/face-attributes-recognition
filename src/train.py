from typing import Dict, Optional

import click
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torchvision.transforms as transforms
from dotenv import load_dotenv
from torch import Tensor

from data.lightning_dm import UtkFaceDataModule
from model.face_features_extractor import FaceFeaturesExtractor, optimizers
from model.backbones import backbone_getters


def get_default_accelerator() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


class CustomModelCheckpoint(ModelCheckpoint):

    def __init__(self, dir_path: str, save_top_k: int, backbone: str, output_layer_depth: int) -> None:
        super().__init__(dirpath=dir_path, save_top_k=save_top_k)
        self.backbone = backbone
        self.output_layer_depth = output_layer_depth

    def format_checkpoint_name(
            self, metrics: Dict[str, Tensor], filename: Optional[str] = None, ver: Optional[int] = None
    ) -> str:
        return super().format_checkpoint_name(
            metrics, f"{self.backbone}-{self.output_layer_depth}-{{val_loss}}", ver
        )


@click.command()
@click.option("--output-layer-depth", type=int, default=4)
@click.option("--backbone", type=click.Choice(backbone_getters), default="mobilenet-v3")
@click.option("--data-dir", type=click.Path(exists=False, file_okay=False), default="../data/")
@click.option("--batch-size", type=int, default=32)
@click.option("--accelerator", type=click.Choice(["cpu", "gpu", "tpu"]), default=get_default_accelerator())
@click.option("--precision", type=click.Choice([
    "transformer-engine", "transformer-engine-float16", "16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true",
    "64-true",
]), default=None)
@click.option("--epochs", type=click.IntRange(1, 1000), default=5)
@click.option("--learning-rate", "--lr", type=float, default=1e-3)
@click.option("--num-workers", type=int, default=2)
@click.option("--models-dir", type=click.Path(exists=True, file_okay=False), default="../models")
@click.option("--optimizer", type=click.Choice(optimizers), default="adam")
def main(
        output_layer_depth: int,
        backbone: str,
        data_dir: str,
        batch_size: int,
        accelerator: str,
        precision: str,
        epochs: int,
        learning_rate: float,
        num_workers: int,
        models_dir: str,
        optimizer: str
) -> None:
    load_dotenv()
    logger = TensorBoardLogger("../runs", name="face-features-extraction")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    model = FaceFeaturesExtractor(backbone_getters[backbone], learning_rate, optimizer, output_layer_depth)
    data_module = UtkFaceDataModule(data_dir, batch_size, num_workers, transform=transform)
    trainer = Trainer(
        accelerator=accelerator,
        precision=precision,
        min_epochs=1,
        max_epochs=epochs,
        logger=logger,
        callbacks=[
            CustomModelCheckpoint(models_dir, save_top_k=1, backbone=backbone, output_layer_depth=output_layer_depth)
        ]
    )
    trainer.fit(model, data_module)
    val_steps_metrics = trainer.validate(model, data_module)
    val_age_loss, val_gender_loss, val_race_loss, val_avg_loss = 0, 0, 0, 0
    for metrics in val_steps_metrics:
        val_age_loss += metrics["age_loss"]
        val_gender_loss += metrics["gender_loss"]
        val_race_loss += metrics["race_loss"]
        val_avg_loss += metrics["loss"]
    metrics_count = len(val_steps_metrics)
    logger.log_hyperparams(
        params={
            "output_layer_depth": output_layer_depth,
            "backbone": backbone,
            "batch_size": batch_size,
            "precision": precision,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer": optimizer
        },
        metrics={
            "val_age_loss": val_age_loss / metrics_count,
            "val_gender_loss": val_gender_loss / metrics_count,
            "val_race_loss": val_race_loss / metrics_count,
            "val_avg_loss": val_avg_loss / metrics_count,
        }
    )


if __name__ == '__main__':
    main()
