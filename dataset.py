from torch.utils.data import Dataset
from utils import normalize_dataset

class EMDataset(Dataset):
    def __init__(self, dataset, normalize=False):
        self.dataset = dataset.drop_duplicates()
        self.normalize = normalize

        if self.normalize:
            self.dataset = normalize_dataset(
                self.dataset, "data/normalization_info.json",
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.iloc[idx].values
