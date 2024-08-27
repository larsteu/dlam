import argparse
import torch
from torch.utils.data import DataLoader
from models.model import EMModel
from dataset import DatasetWithoutLeagues
from utils import load_dataset, preprocess_dataset, plot_loss, plot_accuracy
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Dataset properties ##
TRAIN_DATASET_PATH = [
    "./data/bundesliga_avg_data.csv",
    "./data/ligue1_16-23.csv",
    "./data/la_liga_16-23.csv",
    "./data/serie_a_16-23.csv",
    "./data/premier_league_avg_data.csv",
]
TEST_DATASET_PATH = []
MAPPINGS_FILE_PATH_TRAIN = "data/mappings_without_names_train_model1.json"
MAPPINGS_FILE_PATH_TEST = "data/mappings_without_names_test_model1.json"
CATEGORICAL_COLUMNS = ["home/away", "player_name", "player_position"]
DROP_COLUMNS = ["game_won", "rating"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the base EM Model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_path", type=str, default="./trained_models/base_model", help="Path to save/load the model")  # fmt: skip
    parser.add_argument("--load_model", action="store_true", help="Load a pre-trained model")
    return parser.parse_args()


def train(args):
    full_dataset = load_dataset(TRAIN_DATASET_PATH)
    full_dataset = preprocess_dataset(
        full_dataset,
        CATEGORICAL_COLUMNS,
        MAPPINGS_FILE_PATH_TRAIN,
        DROP_COLUMNS,
        remove_player_names=True,
    )

    # Create the full dataset
    full_dataset = DatasetWithoutLeagues(full_dataset, normalize=True)

    # Get the number of samples in the dataset
    dataset_size = len(full_dataset)

    # Create train/validation split
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    # Use random_split to create the split
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    data_loader_train = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    data_loader_val = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,  # No need to shuffle validation data
    )

    em_model = EMModel().to(DEVICE)

    optimizer = torch.optim.Adam(em_model.parameters(), lr=args.lr)

    best_eval_loss = None
    model_path = Path(args.model_path)

    # Save the accuracies and losses for plotting with their respective epochs
    save_accuracies_eval = {}
    save_losses_eval = {}

    if args.load_model and model_path.exists():
        em_model.load_model(optimizer, args.lr, model_path)
        best_eval_loss = em_model.eval_model(dataloader=data_loader_val, device=DEVICE)
        print(f"Loaded pre-trained model. Initial evaluation loss: {best_eval_loss}")

    for num_epoch in range(args.epochs):
        curr_loss, curr_acc = em_model.train_epoch(
            epoch_idx=num_epoch,
            dataloader=data_loader_train,
            optimizer=optimizer,
            device=DEVICE,
        )

        print(f"Epoch {num_epoch + 1}/{args.epochs}, Training Loss: {curr_loss}", f"Accuracy: {curr_acc}")

        curr_eval_loss, curr_eval_accuracy = em_model.eval_model(dataloader=data_loader_val, device=DEVICE)
        print(f"Epoch {num_epoch + 1}/{args.epochs}, Evaluation Loss: {curr_eval_loss}", f"Accuracy: {curr_eval_accuracy}")

        # Save the accuracies and losses for plotting with their respective epochs
        save_accuracies_eval[num_epoch] = curr_eval_accuracy
        save_losses_eval[num_epoch] = curr_eval_loss

        if best_eval_loss is None or curr_eval_loss < best_eval_loss:
            em_model.save_model(optimizer, model_path)
            best_eval_loss = curr_eval_loss
            print(f"Model saved. New best evaluation loss: {best_eval_loss}")

    plot_loss(None, save_losses_eval, title="Loss")
    plot_accuracy(None, save_accuracies_eval, title="Accuracy")
    print(f"Training completed. Final best evaluation loss: {best_eval_loss}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
