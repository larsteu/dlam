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
            nn.Linear(8, 2),
            nn.Sigmoid(),
        )

    def forward(self, x, weight_1=1, weight_2=1):
        x_1 = x[:, :, :26]
        x_2 = x[:, :, 26:]
        x_1 = self.teamClassifier(x_1) * weight_1
        x_2 = self.teamClassifier(x_2) * weight_2
        teams = torch.concat((x_1, x_2), dim=1)
        return self.gameClassifier(teams)

    def custom_loss(self, outputs, targets, draw_threshold=0.05, mse_weight=0.7, outcome_weight=0.3):
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

    def train_epoch(self, epoch_idx, dataloader, optimizer, device):
        loop = tqdm(dataloader)
        loop.set_description(f"EM Model - Training epoch {epoch_idx}")
        loss_fn = self.get_loss()
        mean_loss = []
        for i, data in enumerate(loop):
            inputs, target = data
            inputs = inputs.float().to(device)
            target = target.float().to(device)

            outputs = self(inputs)

            # TODO: possibility 1
            '''
            # compute difference between the two teams and give this to the loss function
            output_diff = outputs[:, 0] - outputs[:, 1]
            target_diff = target[:, 0] - target[:, 1]

            loss = loss_fn(output_diff, target_diff)
            
            # TODO: possibility 2
            # loss = loss_fn(outputs[:, 0], target[:, 0]) + loss_fn(outputs[:, 1], target[:, 1])

            # TODO: possibility 3
            # Get the prediction correctness tensor
            result_tensor = calculate_correct_predictions(outputs, target, 0.05, return_tensor=True).to(device)

            # Create a weight tensor (2 for incorrect predictions, 1 for correct ones)
            weight_tensor = 3 - result_tensor  # This will be 2 for incorrect and 1 for correct predictions

            # Calculate the element-wise loss
            element_wise_loss = loss_fn(outputs, target)

            # Apply the weights to the loss
            loss = (element_wise_loss * weight_tensor.unsqueeze(1)).mean()
            '''

            loss = self.custom_loss(outputs, target)

            loss.backward()
            mean_loss.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()

            loop.set_postfix({"Loss": torch.tensor(mean_loss).mean().item()})

        return torch.tensor(mean_loss).mean().item()

    def eval_model(self, dataloader, device):
        loop = tqdm(dataloader)
        loop.set_description(f"EM Model - Evaluation")

        mean_loss = []
        loss_fn = self.get_loss()

        correct_predictions = 0
        total_predictions = 0

        self.train(False)

        # print the prediction and target values for the first batch
        first = True
        with torch.no_grad():
            for i, data in enumerate(loop):
                inputs, target = data
                inputs = inputs.float().to(device)
                target = target.float().to(device)
                outputs = self(inputs)
                # TODO: possibility 1
                '''
                # compute difference between the two teams and give this to the loss function
                output_diff = outputs[:, 0] - outputs[:, 1]
                target_diff = target[:, 0] - target[:, 1]

                loss = loss_fn(output_diff, target_diff)
                
                # TODO: possibility 2
                # Calculate loss
                element_wise_loss = loss_fn(outputs, target)
                loss = element_wise_loss.mean()
                '''

                loss = self.custom_loss(outputs, target)
                mean_loss.append(loss.item())

                if first:
                    print("Prediction:", outputs[0].to("cpu").numpy())
                    print("Target:", target[0].to("cpu").numpy())
                    first = False

                # save the number of correct predictions
                correct, total = calculate_correct_predictions(outputs, target, 0.05)
                correct_predictions += correct
                total_predictions += total

            loop.set_postfix({"Loss": torch.tensor(mean_loss).mean().item()})

        self.train(True)
        avg_loss = torch.tensor(mean_loss).mean().item()
        accuracy = correct_predictions / total_predictions

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
        return nn.MSELoss(reduction='none')


class TeamBlock(nn.Module):
    def __init__(self, features=16, output=32):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=(1, 4), stride=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(features),
            nn.Conv2d(features, 2 * features, kernel_size=(1, 4), stride=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(2 * features),
            nn.Conv2d(2 * features, 4 * features, kernel_size=(1, 4), stride=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(4 * features),
            nn.Conv2d(4 * features, 8 * features, kernel_size=(4, 1), stride=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(8 * features),
            nn.Conv2d(8 * features, 16 * features, kernel_size=(4, 1), stride=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16 * features),
            nn.Conv2d(16 * features, 32 * features, kernel_size=(4, 1), stride=(2, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(32 * features),
        )
        self.fc_block = nn.Sequential(
            nn.Linear(32 * features, 16 * features),
            nn.ReLU(),
            nn.Linear(16 * features, 8 * features),
            nn.ReLU(),
            nn.Linear(8 * features, 4 * features),
            nn.ReLU(),
            nn.Linear(4 * features, output),
            nn.ReLU(),
        )

    def forward(self, x):
        x_1 = self.conv_block(x)
        x_2 = x_1.view(x_1.shape[0], -1)
        return self.fc_block(x_2)
