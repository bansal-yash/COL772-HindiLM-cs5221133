# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List

from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel

# You can also create additional files in this directory and import them here if needed.
# For example, the line below import a dummy function from utils.py file.
from .utils import dummy_function  # Replace with actual utility functions as needed

# You can structure your code as you see fit as long as the CLI works as specified.
# Finally, treat this as your FINAL MODEL TRAINING SCRIPT. Do not perform hyperparameter tuning here.
# You can create separate scripts for hyperparameter tuning if needed.

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# All the Hyperparameters of the model
CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "d_head": 32,
    "n_layers": 8,
    "mode": "standard",
}

TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 128
LR = 1e-4
NUM_EPOCHS = 10


class LMDataset(Dataset):
    def __init__(self, corpus, tokenizer):
        self.data = []

        for sentence in tqdm(corpus):
            tokens = tokenizer.encode(sentence)
            if len(tokens) > 1:
                self.data.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = self.data[index]
        attention_mask = torch.ones_like(input_ids)

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        return batch


def new_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    PAD_ID = 0  # Assume 0 is the padding token ID

    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]
    max_seq_len = max(input_ids.shape[0] for input_ids in input_ids_list)

    batch_input_ids = []
    batch_attention_mask = []

    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        pad_len = max_seq_len - input_ids.shape[0]

        padded_input_ids = torch.cat(
            [
                input_ids,
                torch.full(
                    (pad_len,), PAD_ID, dtype=input_ids.dtype, device=input_ids.device
                ),
            ]
        )

        padded_attention_mask = torch.cat(
            [
                attention_mask,
                torch.zeros(
                    pad_len, dtype=attention_mask.dtype, device=attention_mask.device
                ),
            ]
        )

        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(padded_attention_mask)

    collated_batch = {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
    }

    return collated_batch


def train_model(model, train_loader, val_loader, optimizer, criterion, save_path):
    os.makedirs(save_path, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):

        # Training
        model.train()
        total_train_loss = 0

        train_bar = tqdm(
            train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training"
        )

        for batch in train_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            train_bar.set_postfix(avg_loss=total_train_loss / (train_bar.n + 1))

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation")

        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                logits = model(input_ids, attention_mask)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                total_val_loss += loss.item()

                val_bar.set_postfix(avg_loss=total_val_loss / (val_bar.n + 1))

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # Saving the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": CONFIG,
            }

            torch.save(checkpoint, os.path.join(save_path, "best_model.pt"))

            print(f"Best Model saved with val loss: {best_val_loss:4f}")

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": CONFIG,
    }

    torch.save(final_checkpoint, os.path.join(save_path, "final_model.pt"))

    print(f"Overall Best checkpoint val loss: {best_val_loss:4f}")

    # Plot the losses
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")

    plot_path = os.path.join(save_path, "loss_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Loss plot saved at {plot_path}")


def main(args):
    # Load the training and validation corpus
    train_corpus = []
    with open(args.train_path, "r", encoding="utf-8") as f:
        for line in f:
            train_corpus.append(line.strip())

    print(f"Loaded {len(train_corpus)} training sentences from the dataset.")

    val_corpus = []
    with open(args.valid_path, "r", encoding="utf-8") as f:
        for line in f:
            val_corpus.append(line.strip())

    print(f"Loaded {len(val_corpus)} validation sentences from the dataset.")

    # Loading the saved tokenizer
    tokenizer = BPETokenizer(vocab_size=None)
    tokenizer.load(args.tokenizer_path)

    CONFIG["vocab_size"] = tokenizer.get_vocab_size()
    print(CONFIG)

    print("Creating datasets and dataloaders")
    train_dataset = LMDataset(train_corpus, tokenizer)
    val_dataset = LMDataset(val_corpus, tokenizer)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=new_collate_fn,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=new_collate_fn,
    )

    # Constructing model with given configs
    model = LanguageModel(config=CONFIG).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    print("Training the model")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        save_path=args.output_model_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model on the given dataset.")
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to the train dataset"
    )
    parser.add_argument(
        "--valid_path", type=str, required=True, help="Path to the valid dataset"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the tokenizer"
    )
    parser.add_argument(
        "--output_model_path",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()
    main(args)
