import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from utils import calculate_correct_predictions
import torch.nn.functional as functional


class EMModel(nn.Module):
    def __init__(self, team_dim=32):
        super().__init__()
        self.teamClassifier = TeamBlock()
        self.gameClassifier = nn.Sequential(
            nn.Linear(2 * team_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x, weight_1=1, weight_2=1):
        x_1 = x[:, :, :26]
        x_2 = x[:, :, 26:]
        x_1 = self.teamClassifier(x_1) * weight_1
        x_2 = self.teamClassifier(x_2) * weight_2
        teams = torch.concat((x_1, x_2), dim=1)
        x_3 = self.gameClassifier(teams)

        return x_3

    def train_epoch(self, epoch_idx, dataloader, optimizer, device):
        """
        Expects the model to be in training mode.
        """
        loop = tqdm(dataloader)
        loop.set_description(f"EM Model - Training epoch {epoch_idx}")
        loss_fn = self.get_loss()
        mean_loss = []
        correct_predictions = 0
        total_predictions = 0

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

            # save the number of correct predictions
            correct, total = calculate_correct_predictions(outputs, target)
            correct_predictions += correct
            total_predictions += total

            loop.set_postfix({"Loss": torch.tensor(mean_loss).mean().item()})

        accuracy = correct_predictions / total_predictions
        return torch.tensor(mean_loss).mean().item(), accuracy

    def eval_model(self, dataloader, device):
        """
        Expects the model to be in evaluation mode.
        """
        loop = tqdm(dataloader)
        loop.set_description("EM Model - Evaluation")

        mean_loss = []
        loss_fn = self.get_loss()

        correct_predictions = 0
        total_predictions = 0

        self.eval()

        with torch.no_grad():
            for i, data in enumerate(loop):
                inputs, target = data
                inputs = inputs.float().to(device)
                target = target.float().to(device)
                with torch.no_grad():
                    outputs = self(inputs)
                    loss = loss_fn(outputs, target)

                mean_loss.append(loss.item())

                # save the number of correct predictions
                correct, total = calculate_correct_predictions(outputs, target)
                correct_predictions += correct
                total_predictions += total

            loop.set_postfix({"Loss": torch.tensor(mean_loss).mean().item()})

        avg_loss = torch.tensor(mean_loss).mean().item()
        accuracy = correct_predictions / total_predictions

        self.train()
        return avg_loss, accuracy

    def save_model(self, optimizer, path: Path):
        print("=> Saving checkpoint")
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # Create the directory if it does not exist
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, path)

    def load_model(self, optimizer, lr, path: Path):
        print("=> Loading checkpoint")
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def get_loss(self):
        return nn.CrossEntropyLoss()


class TeamBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Linear(572, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )


    def forward(self, x):
        x_1 = x.view(x.shape[0], -1)
        x_2 = self.conv_block(x_1)
        return x_2
