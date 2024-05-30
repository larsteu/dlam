import torch
import torch.nn as nn
from tqdm import tqdm


class EMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.teamClassifier1 = TeamBlock()
        self.teamClassifier2 = TeamBlock()
        self.gameClassifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x, weight_1=1, weight_2=1):
        x_1 = x[:, :26].view(x.shape[0], -1)
        x_2 = x[:, 26:].view(x.shape[0], -1)
        x_1 = self.teamClassifier1(x_1) * weight_1
        x_2 = self.teamClassifier1(x_2) * weight_2
        teams = torch.concat((x_1, x_2), dim=1)
        return self.gameClassifier(teams)

    def train_epoch(self, epoch_idx, dataloader, loss_fn, optimizer, device):
        loop = tqdm(dataloader)
        loop.set_description(f"EM Model - Training epoch {epoch_idx}")
        mean_loss = []

        for i, data in enumerate(loop):
            inputs, target = data
            inputs = inputs.float().to(device)
            target = target.float().to(device)

            outputs = self(inputs)

            loss = loss_fn(outputs, target)
            loss.backward()
            mean_loss.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0:
                loop.set_postfix({"Loss": loss.item()})

    def get_loss(self):
        return nn.MSELoss()


class TeamBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(624, 312),
            nn.ReLU(),
            nn.Linear(312, 156),
            nn.ReLU(),
            nn.Linear(156, 78),
            nn.ReLU(),
            nn.Linear(78, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
