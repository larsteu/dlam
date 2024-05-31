import pandas as pd
from torch.utils.data import Dataset
from utils import normalize_dataset
import numpy as np


class EMDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, normalize=False):
        self.dataset = dataset
        self.normalize = normalize
        self.dataset["team_1_goals"] = self.dataset["game_result"].map(lambda x: int(x.split(sep="–")[0]))
        self.dataset["team_2_goals"] = self.dataset["game_result"].map(lambda x: int(x.split(sep="–")[1]))
        self.dataset = dataset.drop(columns="game_result")
        if self.normalize:
            self.dataset = normalize_dataset(
                self.dataset, "data/normalization_info.json",
            )

    def __len__(self):
        return int(len(self.dataset) / 52)

    def __getitem__(self, idx):
        label_1 = self.dataset.iloc[idx*52].values[-2]
        label_2 = self.dataset.iloc[idx*52].values[-1]
        data = self.dataset.drop(columns=["team_1_goals", "team_2_goals"])
        data = data.iloc[idx * 52:(idx * 52) + 52].values
        return np.array([data]), np.array([label_1, label_2])
