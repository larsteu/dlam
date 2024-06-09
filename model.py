import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class EMModel(nn.Module):
    def __init__(self, team_dim=32, num_leagues=10):
        super().__init__()
        self.teamClassifier = TeamBlock()
        self.gameClassifier = nn.Sequential(
            nn.Linear(2*team_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )

        self.train_with_weights = False

        # We will input the league of each player. Each league will be mapped to a learnable weight and then the team embeddings will be multiplied by the average of the league weights of the players in the team
        self.league_weights = nn.Embedding(num_leagues+1, 1) # Have one extra league for leagues not in the num_leagues

    def forward(self, x):
        x_1 = x[:, :, :26]
        x_2 = x[:, :, 26]

        if self.train_with_weights:
            # TODO: this is just a rough idea of how we get the weights here
            # league info for each player could be the last entry in the input tensor.
            leagues_team_1 = x_1[:, :, -1]
            leagues_team_2 = x_2[:, :, -1]
            x_1 = x_1[:, :, :-1]
            x_2 = x_2[:, :, :-1]

        x_1 = self.teamClassifier(x_1)
        x_2 = self.teamClassifier(x_2)

        if self.train_with_weights:
            # TODO: check if this is works. Not sure, just my first idea
            # Calculate the average league weights for each team
            league_weights_team_1 = self.league_weights(leagues_team_1)
            league_weights_team_2 = self.league_weights(leagues_team_2)
            league_weights_team_1 = league_weights_team_1.mean(dim=1)
            league_weights_team_2 = league_weights_team_2.mean(dim=1)

            x_1 = x_1 * league_weights_team_1
            x_2 = x_2 * league_weights_team_2
            
        teams = torch.concat((x_1, x_2), dim=1)
        return self.gameClassifier(teams)

    def use_weights(self, use_weights):
        self.train_with_weights = use_weights

        # Freeze parameters of the team and game classifiers
        for param in self.teamClassifier.parameters():
            param.requires_grad = use_weights
        for param in self.gameClassifier.parameters():
            param.requires_grad = use_weights

    def train_epoch_to_determine_weights(self, epoch_idx, dataloader, loss_fn, optimizer, device):
        # 1. Freeze parameters of the team and game classifiers
        self.use_weights(True)

        # 2. Train the model to determine the weights

        # Loop through the dataloader
        # For each batch, append the leagues to the input tensor
        # Everything else should likely be the same as train_epoch

        # 3. Set back to normal mode
        self.use_weights(False)

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
            loop.set_postfix({"Loss": np.array(mean_loss).mean()})

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
    def __init__(self, features=16, output=32):
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
            nn.Linear(4*features, output),
            nn.ReLU()
        )

    def forward(self, x):
        x_1 = self.conv_block(x)
        x_2 = x_1.view(x_1.shape[0], -1)
        return self.fc_block(x_2)
