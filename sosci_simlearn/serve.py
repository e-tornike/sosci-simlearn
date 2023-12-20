import os
import typer

import torch
from quaterion_models.model import SimilarityModel
from quaterion.distances import Distance


def main(
    # test_path: str = "",
    model_path: str = "",
    ):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SimilarityModel.load(os.path.join(model_path, "servable"))
    model.to(device)

    print(model.encode("this is also a query"))

    # test_data = load_jsonl(test_path)
    # test_inputs = []
    
    # test_embeddings = model.encode(test_data)


if __name__ == "__main__":
    typer.run(main)