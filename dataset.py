from torch.utils.data import Dataset
from utils import normalize_dataset
import numpy as np


class EMDataset(Dataset):
    def __init__(self, dataset, normalize=False):
        self.dataset = dataset
        self.normalize = normalize

        if self.normalize:
            self.dataset = normalize_dataset(
                self.dataset, "data/normalization_info.json",
            )

    def __len__(self):
        return int(len(self.dataset) / 52)

    def __getitem__(self, idx):
        label = self.dataset.iloc[idx*52].values[-1]
        data = self.dataset.drop(columns="game_won")
        data = data.iloc[idx * 52:(idx * 52) + 52].values
        return data, np.array([label])
