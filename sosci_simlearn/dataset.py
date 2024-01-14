import random
import jsonlines
from typing import List, Dict
from torch.utils.data import Dataset
from quaterion.dataset.similarity_samples import SimilarityPairSample


class SoSciDataset(Dataset):
    """Dataset class to process .jsonl files with FAQ from popular cloud providers."""

    def __init__(self, dataset_path, obj_a, obj_b):
        self.dataset: List[Dict[str, str]] = self.read_dataset(dataset_path)
        self.obj_a = obj_a
        self.obj_b = obj_b

    def __getitem__(self, index) -> SimilarityPairSample:
        line = self.dataset[index]
        sentence_1 = line[self.obj_a]
        sentence_2 = line[self.obj_b]
        subgroup = hash(sentence_1+sentence_2)
        return SimilarityPairSample(
            obj_a=sentence_1, obj_b=sentence_2, score=1, subgroup=subgroup
        )

    def __len__(self):
        return len(self.dataset)
    
    def sample(self, n):
        self.dataset = random.sample(self.dataset, n)

    @staticmethod
    def read_dataset(dataset_path) -> List[Dict[str, str]]:
        """Read jsonl-file into a memory."""
        data = []
        with jsonlines.open(dataset_path, "r") as reader:
            for obj in reader:
                data.append(obj)
        return data