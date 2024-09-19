import torch
import torch.nn as nn
from tqdm import tqdm
from models.model import EMModel


class LeagueEmbedding(nn.Module):
    def __init__(self, num_leagues, embedding_dim=4):
        super().__init__()
        self.num_leagues = num_leagues
        self.embedding = nn.Embedding(num_leagues + 1, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, league_ids):
        league_ids = torch.clamp(league_ids, 0, self.num_leagues)
        embedded = self.embedding(league_ids)
        weights = self.fc(embedded)
        return weights.squeeze(-1)


class EMModelWithLeague(EMModel):
    def __init__(self, num_leagues, team_dim=32):
        super().__init__(team_dim)
        self.league_embedding = LeagueEmbedding(num_leagues)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, league_ids):
        league_weights = self.league_embedding(league_ids)

        x_1 = x[:, :, :26].float()
        x_2 = x[:, :, 26:].float()

        league_weights_1 = league_weights[:, :26].unsqueeze(1).unsqueeze(-1)
        league_weights_2 = league_weights[:, 26:].unsqueeze(1).unsqueeze(-1)

        x_1 = x_1 * league_weights_1
        x_2 = x_2 * league_weights_2

        x_1 = self.teamClassifier(x_1)
        x_2 = self.teamClassifier(x_2)

        x_1 = self.dropout(x_1)
        x_2 = self.dropout(x_2)

        teams = torch.cat((x_1, x_2), dim=1)
        teams = teams.view(teams.size(0), -1)

        output = self.gameClassifier(teams)

        return output

    def train_epoch(self, epoch_idx, dataloader, loss_fn, optimizer, device):
        self.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for inputs, league_ids, targets in dataloader:
            inputs, league_ids, targets = inputs.to(device), league_ids.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = self(inputs, league_ids)
            loss = loss_fn(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(targets, 1)
            correct_predictions += (predicted == true_labels).sum().item()
            total_predictions += targets.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def eval_model(self, dataloader, device):
        self.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        loss_fn = self.get_loss()

        with torch.no_grad():
            for inputs, league_ids, targets in dataloader:
                inputs, league_ids, targets = inputs.to(device), league_ids.to(device), targets.to(device)

                outputs = self(inputs, league_ids)
                loss = loss_fn(outputs, targets)

                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(targets, 1)
                correct_predictions += (predicted == true_labels).sum().item()
                total_predictions += targets.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions

        return avg_loss, accuracy

    def freeze_base_model(self):
        for param in self.teamClassifier.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        for param in self.teamClassifier.parameters():
            param.requires_grad = True