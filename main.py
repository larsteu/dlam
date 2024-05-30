import torch
from torch.utils.data import DataLoader

from model import EMModel
from dataset import EMDataset
from utils import load_dataset, preprocess_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 10


def train():
    dataset_train = load_dataset()
    dataset_train = preprocess_dataset(dataset_train)

    dataset = EMDataset(dataset_train, normalize=True)
    data_loader = DataLoader(dataset,
                             batch_size=64,
                             shuffle=True)

    em_model = EMModel().to(DEVICE)

    optimizer = torch.optim.Adam(em_model.parameters(), lr=1e-5)
    loss_fn = em_model.get_loss()

    for num_epoch in range(NUM_EPOCHS):
        em_model.train_epoch(
            epoch_idx=num_epoch,
            dataloader=data_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=DEVICE)


if __name__ == '__main__':
    train()
