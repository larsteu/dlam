import torch
from torch.utils.data import DataLoader

from model import EMModel
from dataset import EMDataset

NUM_EPOCHS = 10


def train():
    dataset = EMDataset()
    data_loader = DataLoader(dataset,
                             batch_size=64,
                             shuffle=True)
    em_model = EMModel()
    optimizer = torch.optim.Adam(em_model.parameters(), lr=1e-5)
    loss_fn = em_model.get_loss()

    for num_epoch in range(NUM_EPOCHS):
        em_model.train_epoch(
            epoch_idx=num_epoch,
            dataloader=data_loader,
            loss_fn=loss_fn,
            optimizer=optimizer)


if __name__ == '__main__':
    train()
