import os
import typer

import pandas as pd
from sklearn.model_selection import train_test_split


def main(
    data_path: str = "/home/tornike/Coding/phd/sosci-simlearn/data/filtered_meta_pairs_20240113-122949.jsonl",    
    seed: int = 42,
    test_size: float = 0.1,
    ):
    assert os.path.isfile(data_path)
    df = pd.read_json(data_path, lines=True)
    train_X, val_X = train_test_split(
        df, 
        stratify=df.source, 
        random_state=seed, 
        test_size=test_size
    )

    train_X.to_json(
        data_path.replace(".json", "_train.json"),
        orient="records",
        lines=True,
    )
    val_X.to_json(
        data_path.replace(".json", "_val.json"),
        orient="records",
        lines=True,
    )


if __name__ == "__main__":
    typer.run(main)