# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import math
from multiprocessing import Pool

from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel, MAX_SEQ_LEN

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
    "d_model": 2048,
    "n_heads": 16,
    "d_head": 128,
    "n_layers": 8,
    "mode": "standard",
}

TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 16
LR = 1e-4
NUM_EPOCHS = 10


def encode_sentence(args):
    sentence, tokenizer = args
    tokens = tokenizer.encode(sentence)
    return tokens


class LMDataset(Dataset):
    def __init__(self, corpus, tokenizer):
        self.seq_len = MAX_SEQ_LEN
        self.all_tokens = []

        with Pool(processes=os.cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        encode_sentence, [(s, tokenizer) for s in corpus], chunksize=32
                    ),
                    total=len(corpus),
                )
            )

        for tokens in results:
            self.all_tokens.extend(tokens)

        self.reshuffle()

    def reshuffle(self):
        offset = random.randint(0, self.seq_len - 1)
        tokens = self.all_tokens[offset:]
        self.chunks = []
        for i in range(0, len(tokens) - self.seq_len, self.seq_len):
            chunk = tokens[i : i + self.seq_len + 1]
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        chunk = self.chunks[index]
        input_ids = chunk[:-1]
        labels = chunk[1:]
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def new_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, criterion, save_path
):
    os.makedirs(save_path, exist_ok=True)

    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []

    best_val_loss = float("inf")
    scaler = torch.amp.GradScaler(device.type)

    for epoch in range(NUM_EPOCHS):

        # Training
        model.train()
        train_loader.dataset.reshuffle()
        total_train_loss = 0

        train_bar = tqdm(
            train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Training"
        )

        for batch in train_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device.type):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

            train_bar.set_postfix(avg_loss=total_train_loss / (train_bar.n + 1))

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_train_ppl = math.exp(avg_train_loss)
        train_ppls.append(avg_train_ppl)

        scheduler.step()
        torch.cuda.empty_cache()

        # Validation
        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation")

        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.amp.autocast(device.type):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                total_val_loss += loss.item()

                val_bar.set_postfix(avg_loss=total_val_loss / (val_bar.n + 1))

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        avg_val_ppl = math.exp(avg_val_loss)
        val_ppls.append(avg_val_ppl)

        print(
            f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train PPL: {avg_train_ppl:.2f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val PPL: {avg_val_ppl:.2f}"
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

            print(f"Best Model saved with val loss: {best_val_loss:4f}\n")

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

    # Plot the perplexities
    plt.figure()
    plt.plot(range(1, NUM_EPOCHS + 1), train_ppls, label="Train Perplexity")
    plt.plot(range(1, NUM_EPOCHS + 1), val_ppls, label="Val Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.title("Training & Validation Perplexity")

    ppl_plot_path = os.path.join(save_path, "perplexity_plot.png")
    plt.savefig(ppl_plot_path)
    plt.close()

    print(f"Perplexity plot saved at {ppl_plot_path}")


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
    tokenizer = BPETokenizer()
    tokenizer_loaded = tokenizer.load(args.tokenizer_path)

    if not tokenizer_loaded:
        print("Please train the tokenizer before loading and training the model.")
        return

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
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=new_collate_fn,
    )

    # Constructing model with given configs
    model = LanguageModel(config=CONFIG).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1_000_000:.2f} M")

    decay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    no_decay_params = [
        p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LR,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    print("Training the model")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
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
