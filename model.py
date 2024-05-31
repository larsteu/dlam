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
            nn.Linear(8, 2),
            nn.Sigmoid()
        )

    def forward(self, x, weight_1=1, weight_2=1):
        x_1 = x[:, :, :26]
        x_2 = x[:, :, 26:]
        x_1 = self.teamClassifier1(x_1) * weight_1
        x_2 = self.teamClassifier2(x_2) * weight_2
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
            mean_loss.append(loss.to("cpu").item())

            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0:
                loop.set_postfix({"Loss": loss.to("cpu").item()})

    def save_model(self, optimizer, path):
        print("=> Saving checkpoint")
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(checkpoint, path)

    def load_model(self, optimizer, lr, path):
        print("=> Loading checkpoint")
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def get_loss(self):
        return nn.MSELoss()


class TeamBlock(nn.Module):
    def __init__(self, features=16):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=(1, 4), stride=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(features),

            nn.Conv2d(features, 2*features, kernel_size=(1, 4), stride=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(2*features),

            nn.Conv2d(2*features, 4*features, kernel_size=(1, 4), stride=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(4*features),

            nn.Conv2d(4*features, 8*features, kernel_size=(4, 1), stride=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(8*features),

            nn.Conv2d(8*features, 16*features, kernel_size=(4, 1), stride=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16*features),

            nn.Conv2d(16*features, 32*features, kernel_size=(4, 1), stride=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32*features),
        )
        self.fc_block = nn.Sequential(
            nn.Linear(32*features, 16*features),
            nn.ReLU(),
            nn.Linear(16*features, 8*features),
            nn.ReLU(),
            nn.Linear(8*features, 4*features),
            nn.ReLU(),
            nn.Linear(4*features, 32),
            nn.ReLU()
        )

    def forward(self, x):
        x_1 = self.conv_block(x)
        x_2 = x_1.view(x_1.shape[0], -1)
        return self.fc_block(x_2)
