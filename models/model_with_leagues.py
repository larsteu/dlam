import torch
import torch.nn as nn
from torch import Tensor
from models.model import EMModel
from tqdm import tqdm
import numpy as np


class LeagueToScalar(nn.Module):
    """
    Produces a single scalar value for representing the team strength depending on the league of the players.
    This scalar is determined by 1) embedding the league IDs and 2) passing the embeddings through a linear layer.

    The LeagueToScalar class reduces the league embeddings to a single scalar value per team, which is then used to scale the entire team embedding tensor.
    This approach might not capture the individual contributions of players from different leagues correctly.
    However, because we only scale the data after the team embeddings have been passed through the team classifier, we do not need to train the teamClassifier but we can freeze it.
    This is necessary because of the low quantity of data available for training step 2: We need to freeze as many parameters as possible.
    """

    def __init__(self, num_leagues, num_players) -> None:
        super().__init__()

        # 1. Have an embedding layer
        self.embedding = nn.Embedding(num_leagues, 1)

        # 2. Have a linear layer
        self.linear = nn.Linear(num_players, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x is a tensor of shape (batch_size, num_players)
        x = self.embedding(x)  # shape: (batch_size, num_players, 1)
        x = x.squeeze(-1)  # shape: (batch_size, num_players)
        x = self.linear(x)  # shape: (batch_size, 1)
        return x


class EMModelWithLeague(EMModel):
    def __init__(self, num_leagues, team_dim=32):
        super().__init__(team_dim)
        self.leagueToScalar = LeagueToScalar(num_leagues, 26)

    def forward(self, x, league_ids):
        # x is a tensor of shape (batch_size, team_dim, 52)
        # league_ids is a tensor of shape (batch_size, 52)
        x_1 = x[:, :, :26]
        x_2 = x[:, :, 26:]
        x_1 = self.teamClassifier(x_1)
        x_2 = self.teamClassifier(x_2)

        # Multiply the team embeddings by the league weights
        league_weights_team_1 = self.leagueToScalar(league_ids[:, :26])
        league_weights_team_2 = self.leagueToScalar(league_ids[:, 26:])
        x_1 = x_1 * league_weights_team_1
        x_2 = x_2 * league_weights_team_2

        # Concatenate the team embeddings & pass through the game classifier
        teams = torch.cat((x_1, x_2), dim=1)
        return self.gameClassifier(teams)

    def train_epoch(self, epoch_idx, dataloader, loss_fn, optimizer, device):
        self.train()
        tqdm_dataloader = tqdm(dataloader)
        tqdm_dataloader.set_description(f"EM Model with League - Training epoch {epoch_idx}")
        mean_loss = []

        for inputs, league_ids, target in tqdm_dataloader:
            # Move data to device
            inputs = inputs.float().to(device)
            league_ids = league_ids.long().to(device)
            target = target.float().to(device)

            # Forward pass, compute loss & backpropagate
            outputs = self(inputs, league_ids)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            mean_loss.append(loss.to("cpu").item())
            tqdm_dataloader.set_postfix({"Loss": np.array(mean_loss).mean()})

    def eval_model(self, dataloader, device):
        self.eval()
        total_loss = 0
        loss_fn = nn.MSELoss()

        with torch.no_grad():
            for inputs, league_ids, target in dataloader:
                inputs = inputs.float().to(device)
                league_ids = league_ids.long().to(device)
                target = target.float().to(device)

                outputs = self(inputs, league_ids)
                loss = loss_fn(outputs, target)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def freeze_team_classifier(self):
        for param in self.teamClassifier.parameters():
            param.requires_grad = False

    def unfreeze_team_classifier(self):
        for param in self.teamClassifier.parameters():
            param.requires_grad = True
