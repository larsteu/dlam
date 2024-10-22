import argparse
import torch
from torch.utils.data import DataLoader
from models.model import EMModel
from dataset import DatasetWithoutLeagues
from training_step_2 import TRAIN_DATASET_PATHS
from utils import load_dataset, preprocess_dataset, plot_loss, plot_accuracy
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## Dataset properties ##

TRAIN_DATASET_PATH = [
    "./data/bundesliga_16-23.csv",
    "./data/ligue1_16-23.csv",
    "./data/la_liga_16-23.csv",
    "./data/serie_a_16-23.csv",
    "./data/premier_league_16-23.csv",
]
'''
TRAIN_DATASET_PATH = [
    "./data/bundesliga_avg_data.csv",
    "./data/bundesliga_16-23.csv",
    "./data/ligue1_16-23.csv",
    "./data/la_liga_avg.csv",
    "./data/serie_a_avg.csv",
    "./data/premier_league_avg_data.csv",
]
'''
TEST_DATASET_PATH = []
MAPPINGS_FILE_PATH_TRAIN = "data/mappings_without_names_train_model1.json"
MAPPINGS_FILE_PATH_TEST = "data/mappings_without_names_test_model1.json"
CATEGORICAL_COLUMNS = ["home/away", "player_name", "player_position"]
DROP_COLUMNS = ["game_won", "rating"]

WORKERS = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Train the base EM Model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
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

    batch_size = 128  # or higher, depending on your GPU memory

    #train_dataset = full_dataset
    #val_dataset = full_dataset

    # Create DataLoaders with more workers and pin_memory
    data_loader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=True,
    )

    data_loader_val = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=WORKERS,
    )

    em_model = EMModel().to(DEVICE)

    optimizer = torch.optim.Adam(em_model.parameters(), lr=args.lr)

    best_eval_accuracy = None
    model_path = Path(args.model_path)
    # Save the accuracies and losses for plotting with their respective epochs
    save_accuracies_eval = {}
    save_losses_eval = {}
    save_accuracies_train = {}
    save_losses_train = {}

    if args.load_model and model_path.exists():
        em_model.load_model(optimizer, args.lr, model_path)
        _, best_eval_accuracy = em_model.eval_model(dataloader=data_loader_val, device=DEVICE)
        print(f"Loaded pre-trained model. Initial evaluation accuracy: {best_eval_accuracy}")

    for num_epoch in range(args.epochs):
        em_model.train()

        curr_loss, curr_acc = em_model.train_epoch(
            epoch_idx=num_epoch,
            dataloader=data_loader_train,
            optimizer=optimizer,
            device=DEVICE,
        )

        save_accuracies_train[num_epoch] = curr_acc
        save_losses_train[num_epoch] = curr_loss

        print(f"Epoch {num_epoch + 1}/{args.epochs}, Training Loss: {curr_loss}", f"Accuracy: {curr_acc}")

        em_model.eval()
        curr_eval_loss, curr_eval_accuracy = em_model.eval_model(dataloader=data_loader_val, device=DEVICE)
        print(
            f"Epoch {num_epoch + 1}/{args.epochs}, Evaluation Loss: {curr_eval_loss},",
            f"Evaluation Accuracy: {curr_eval_accuracy:.2f}, Accuracy on training set (see when it overfits): {curr_acc:.2f}",
        )

        # Save the accuracies and losses for plotting with their respective epochs
        save_accuracies_eval[num_epoch] = curr_eval_accuracy
        save_losses_eval[num_epoch] = curr_eval_loss

        if best_eval_accuracy is None or curr_eval_accuracy > best_eval_accuracy:
            em_model.save_model(optimizer, model_path)
            best_eval_accuracy = curr_eval_accuracy
            print(f"Model saved. New best evaluation accuracy: {best_eval_accuracy}")

    plot_loss(save_losses_train, save_losses_eval, title="Loss")
    plot_accuracy(save_accuracies_train, save_accuracies_eval, title="Accuracy")
    print(f"Training completed. Final best evaluation accuracy: {best_eval_accuracy}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
