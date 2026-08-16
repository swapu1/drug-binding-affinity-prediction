"""
Phase 5: Training Loop (Original / Iteration 1 settings restored)
Batch size 1, plain Adam, no weight decay, no LR scheduler -
this is the exact setup that produced val MSE 2.6232.
Saved to best_model_final.pt this time so it never gets overwritten
by future experiments.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from phase4_model import BindingAffinityModel

GRAPHS_DIR = "graphs"
TRAIN_CSV = "split_train.csv"
VAL_CSV = "split_val.csv"

LEARNING_RATE = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
CHECKPOINT_PATH = "best_model_final.pt"


def load_split_ids(csv_path):
    df = pd.read_csv(csv_path)
    return df["pdb_id"].tolist()


def load_graph(pdb_id):
    path = os.path.join(GRAPHS_DIR, pdb_id + ".pt")
    return torch.load(path)


def run_one_epoch(model, pdb_ids, optimizer, loss_fn, is_training):
    if is_training == True:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    count = 0

    index = 0
    while index < len(pdb_ids):
        pdb_id = pdb_ids[index]
        data = load_graph(pdb_id)

        true_label = torch.tensor(data["label"], dtype=torch.float)

        if is_training == True:
            optimizer.zero_grad()
            prediction = model(
                data["mol_x"], data["mol_edge_index"],
                data["pocket_x"], data["pocket_edge_index"]
            )
            loss = loss_fn(prediction, true_label)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                prediction = model(
                    data["mol_x"], data["mol_edge_index"],
                    data["pocket_x"], data["pocket_edge_index"]
                )
                loss = loss_fn(prediction, true_label)

        total_loss = total_loss + loss.item()
        count = count + 1
        index = index + 1

    average_loss = total_loss / count
    return average_loss


if __name__ == "__main__":
    train_ids = load_split_ids(TRAIN_CSV)
    val_ids = load_split_ids(VAL_CSV)
    print("Train complexes:", len(train_ids))
    print("Val complexes:", len(val_ids))

    model = BindingAffinityModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    epoch = 1
    while epoch <= MAX_EPOCHS:
        train_loss = run_one_epoch(model, train_ids, optimizer, loss_fn, True)
        val_loss = run_one_epoch(model, val_ids, optimizer, loss_fn, False)

        print("Epoch", epoch, "- Train MSE:", round(train_loss, 4), "- Val MSE:", round(val_loss, 4))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("  -> New best val loss, checkpoint saved.")
        else:
            epochs_without_improvement = epochs_without_improvement + 1
            print("  -> No improvement for", epochs_without_improvement, "epoch(s).")

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered at epoch", epoch)
            break

        epoch = epoch + 1

    print("\nTraining complete. Best val MSE:", round(best_val_loss, 4))
    print("Best model saved to", CHECKPOINT_PATH)
