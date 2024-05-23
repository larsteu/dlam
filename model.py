import torch.nn as nn
from tqdm import tqdm


class EMModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def train_epoch(self, epoch_idx, dataloader, loss_fn, optimizer):
        loop = tqdm(dataloader)
        loop.set_description(f"EM Model - Training epoch {epoch_idx}")
        mean_loss = []

        for i, data in enumerate(loop):
            inputs, target = data
            inputs = inputs.float()

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
