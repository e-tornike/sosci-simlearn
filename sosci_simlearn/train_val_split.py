import os
import typer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy


def shuffle_columns(df, column_a="sentence_1", column_b="sentence_2"):
    new_rows = []
    for i in range(df.shape[0]):
        row = deepcopy(df.iloc[i])
        if np.random.rand() < 0.5:
            sent_1 = row[column_a]
            sent_2 = row[column_b]
            row[column_a] = sent_2
            row[column_b] = sent_1
        new_rows.append(row)
    return pd.DataFrame(new_rows)


def main(
    data_path: str = "/home/tornike/Coding/phd/sosci-simlearn/data/filtered_meta_pairs_20240113-122949.jsonl",    
    seed: int = 42,
    test_size: float = 0.075,
    shuffle: bool = True,
    ):
    assert os.path.isfile(data_path)
    df = pd.read_json(data_path, lines=True)
    if shuffle:
        df = shuffle_columns(df)

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