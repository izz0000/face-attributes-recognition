import click
import torch
from PIL import Image
import torchvision.transforms as transforms

import os


@click.command()
@click.argument("model-name", type=str)
@click.argument("image-path", type=click.Path(exists=True, dir_okay=False))
@click.option("--models-dir", type=click.Path(exists=True, file_okay=False))
def main(model_name: str, image_path: str, model_dir: str) -> None:
    assert model_name.endswith(".ckpt"), "model-name should be `ckpt` file"
    model = torch.load(os.path.join(model_dir, model_name))
    print("Starting Predicting...")
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    predictions: torch.Tensor = model(torch.unsqueeze(transform(image), 0))
    age = predictions[0]
    gender_id = predictions[1:3].argmax()[0]
    race_id = predictions[4:].argmax()[1]
    gender = ["Male", "Female"][gender_id]
    race = ["White", "Black", "Asian", "Indian", "Other (Not white, black, asian, or indian)"][race_id]

    print(f"Prediction: \nAge: {age} \nGender: {gender} \nRace: {race}")


if __name__ == '__main__':
    main()
