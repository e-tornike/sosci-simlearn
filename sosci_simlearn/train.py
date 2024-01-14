import os
import typer
import datetime
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from quaterion import Quaterion
from quaterion.dataset import PairsSimilarityDataLoader

from model import SoSciModel
from dataset import SoSciDataset


def train(model, train_dataset_path, val_dataset_path, params, sample_n=0, output_dir=None):
    use_gpu = params.get("cuda", torch.cuda.is_available())

    trainer = pl.Trainer(
        min_epochs=params.get("min_epochs", 1),
        max_epochs=params.get("max_epochs", 300),
        auto_select_gpus=use_gpu,
        log_every_n_steps=params.get("log_every_n_steps", 10),
        gpus=int(use_gpu),
        num_sanity_val_steps=2,
    )
    train_dataset = SoSciDataset(train_dataset_path, obj_a=params.get("obj_a", ""), obj_b=params.get("obj_b", ""))
    val_dataset = SoSciDataset(val_dataset_path, obj_a=params.get("obj_a", ""), obj_b=params.get("obj_b", ""))
    if sample_n > 0:
        train_dataset.sample(sample_n)

    train_dataloader = PairsSimilarityDataLoader(train_dataset, batch_size=params.get("batch_size", 1024))
    val_dataloader = PairsSimilarityDataLoader(val_dataset, batch_size=params.get("batch_size", 1024))
    Quaterion.fit(model, trainer, train_dataloader, val_dataloader)
    if output_dir:
        model.save_servable(os.path.join(output_dir, "servable"))


def main(
    train_path: str = "",
    val_path: str = "",
    seed: int = 42,
    model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2",
    obj_a: str = "sentence_1",
    obj_b: str = "sentence_2",
    batch_size: int = 1024,
    max_epochs: int = 300,
    output_dir: str = "",
    sample_n: int = 0,
    ):
    seed_everything(seed, workers=True)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir, model_name.replace("/", ":")+f"_epochs={max_epochs}_{timestamp}")

    sosci_model = SoSciModel(model_name=model_name)
    train(
        model=sosci_model, 
        train_dataset_path=train_path, 
        val_dataset_path=val_path, 
        params={
            "obj_a": obj_a, 
            "obj_b": obj_b, 
            "batch_size": batch_size,
            "max_epochs": max_epochs,
        },
        sample_n=sample_n,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    typer.run(main)