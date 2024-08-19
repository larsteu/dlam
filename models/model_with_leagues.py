import torch
import torch.nn as nn
from models.model import EMModel
from tqdm import tqdm
import numpy as np


class LeagueEmbedding(nn.Module):
    def __init__(self, num_leagues):
        super().__init__()
        self.embedding = nn.Embedding(num_leagues, 1)

    def forward(self, x):
        return self.embedding(x).squeeze(-1)


class EMModelWithLeague(EMModel):
    def __init__(self, num_leagues, team_dim=32):
        super().__init__(team_dim)
        self.leagueEmbedding = LeagueEmbedding(num_leagues)

    def forward(self, x, league_ids):
        x_1 = x[:, :, :26]
        x_2 = x[:, :, 26:]
        x_1 = self.teamClassifier(x_1)
        x_2 = self.teamClassifier(x_2)

        # Multiply the team embeddings by the league weights
        league_weights = self.leagueEmbedding(league_ids)
        x_1 = x_1 * league_weights[:, 0]
        x_2 = x_2 * league_weights[:, 1]

        # Concatenate the team embeddings & pass through the game classifier
        teams = torch.cat((x_1, x_2), dim=1)
        return self.gameClassifier(teams)

    def train_epoch(self, epoch_idx, dataloader, loss_fn, optimizer, device):
        self.train()
        loop = tqdm(dataloader)
        loop.set_description(f"EM Model with League - Training epoch {epoch_idx}")
        mean_loss = []

        for i, data in enumerate(loop):
            inputs, league, target = data

            # Extract the league ids from the inputs
            league_ids = inputs[:, :, -1].long()

            inputs = inputs.float().to(device)
            target = target.float().to(device)

            outputs = self(inputs, league_ids)

            loss = loss_fn(outputs, target)
            loss.backward()
            mean_loss.append(loss.to("cpu").item())

            optimizer.step()
            optimizer.zero_grad()
            loop.set_postfix({"Loss": np.array(mean_loss).mean()})

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
