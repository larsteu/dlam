from distutils.sysconfig import customize_compiler
from functools import total_ordering

import torch
import torch.nn as nn
from torch import Tensor
from models.model import EMModel
from tqdm import tqdm
import numpy as np
from utils import calculate_correct_predictions
from torch.nn import functional

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

    def custom_loss(self, outputs, targets, draw_threshold=0.05, mse_weight=0.5, outcome_weight=0.5):
        mse_loss = functional.mse_loss(outputs, targets, reduction='none')

        # Calculate match outcome
        pred_diff = outputs[:, 0] - outputs[:, 1]
        true_diff = targets[:, 0] - targets[:, 1]

        pred_outcome = torch.where(torch.abs(pred_diff) < draw_threshold, torch.zeros_like(pred_diff),
                                   torch.sign(pred_diff))
        true_outcome = torch.where(torch.abs(true_diff) < draw_threshold, torch.zeros_like(true_diff),
                                   torch.sign(true_diff))

        # Calculate outcome loss (use binary cross-entropy for 3-class problem)
        outcome_loss = functional.cross_entropy(pred_outcome.unsqueeze(1), true_outcome.float().unsqueeze(1))

        # Combine losses
        combined_loss = mse_weight * mse_loss.mean() + outcome_weight * outcome_loss

        return combined_loss

    def train_epoch(self, epoch_idx, dataloader, loss_fn, optimizer, device, draw_threshold=0.05):
        self.train()
        tqdm_dataloader = tqdm(dataloader)
        tqdm_dataloader.set_description(f"EM Model with League - Training epoch {epoch_idx}")

        correct_predictions = 0
        total_predictions = 0
        mean_loss = []

        for inputs, league_ids, target in tqdm_dataloader:
            # Move data to device
            inputs = inputs.float().to(device)
            league_ids = league_ids.long().to(device)
            target = target.float().to(device)

            # Forward pass, compute loss & backpropagate
            outputs = self(inputs, league_ids)
            '''
            # Get the prediction correctness tensor
            result_tensor = calculate_correct_predictions(outputs, target, draw_threshold, return_tensor=True)

            # Create a weight tensor (2 for incorrect predictions, 1 for correct ones)
            weight_tensor = 4 - result_tensor  # This will be 2 for incorrect and 1 for correct predictions

            # Calculate the element-wise loss
            element_wise_loss = nn.MSELoss(reduction='none')(outputs, target)

            # Apply the weights to the loss
            loss = (element_wise_loss * weight_tensor.unsqueeze(1)).mean()
            '''
            loss = self.custom_loss(outputs, target, draw_threshold)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save the number of correct predictions
            correct, total = calculate_correct_predictions(outputs, target, draw_threshold)
            correct_predictions += correct
            total_predictions += total

            # Logging
            mean_loss.append(loss.to("cpu").item())
            tqdm_dataloader.set_postfix({"Loss": np.array(mean_loss).mean()})

        # calculate the accuracy over the entire dataset
        accuracy = correct_predictions / total_predictions

        return np.array(mean_loss).mean(), accuracy

    def eval_model(self, dataloader, device, draw_threshold=0.05):
        self.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        loss_fn = nn.MSELoss()
        first = True

        with torch.no_grad():
            for inputs, league_ids, target in dataloader:
                inputs = inputs.float().to(device)
                league_ids = league_ids.long().to(device)
                target = target.float().to(device)

                outputs = self(inputs, league_ids)
                #loss = loss_fn(outputs[:, 0], target[:, 0]) + loss_fn(outputs[:, 1], target[:, 1])
                loss = self.custom_loss(outputs, target, draw_threshold)

                total_loss += loss.item()

                if first:
                    print("Prediction:", outputs[0].to("cpu").numpy())
                    print("Target:", target[0].to("cpu").numpy())
                    first = False

                # Calculate the number of correct predictions
                correct, total = calculate_correct_predictions(outputs, target, draw_threshold)
                correct_predictions += correct
                total_predictions += total

        # calculate the accuracy and loss over the entire dataset
        accuracy = correct_predictions / total_predictions
        final_loss = total_loss / len(dataloader)

        return final_loss, accuracy

    def freeze_team_classifier(self):
        for param in self.teamClassifier.parameters():
            param.requires_grad = False

    def unfreeze_team_classifier(self):
        for param in self.teamClassifier.parameters():
            param.requires_grad = True
