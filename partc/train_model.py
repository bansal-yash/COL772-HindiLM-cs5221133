# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
import torch
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel

# You can also create additional files in this directory and import them here if needed.
# For example, the line below import a dummy function from utils.py file.
from .utils import dummy_function  # Replace with actual utility functions as needed

# You can structure your code as you see fit as long as the CLI works as specified.
# Finally, treat this as your FINAL MODEL TRAINING SCRIPT. Do not perform hyperparameter tuning here.
# You can create separate scripts for hyperparameter tuning if needed.

CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "d_head": 32,
    "n_layers": 8,
    "mode": "standard",
}

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(device)


def main(args):
    # Load the training and validation corpus
    train_corupus = []
    with open(args.train_path, "r", encoding="utf-8") as f:
        for line in f:
            train_corupus.append(line.strip())

    print(f"Loaded {len(train_corupus)} training sentences from the dataset.")

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
    # print(train_corupus[0])
    # print(tokenizer.encode(train_corupus[0]))
    # print(tokenizer.decode(tokenizer.encode(train_corupus[0])))

    # Constructing model with given configs
    model = LanguageModel(config=CONFIG).to(device)

    # tokens = tokenizer.encode(train_corupus[0])
    # input_ids = torch.tensor(tokens, dtype=torch.long)
    # input_ids = input_ids.unsqueeze(0).to(device)
    # attention_mask = torch.ones_like(input_ids).to(device)
    # model.eval()
    # with torch.no_grad():
    #     logits = model(input_ids, attention_mask)

    # print(logits)
    # print(logits.shape)
    # predicted_token_ids = torch.argmax(logits, dim=-1)
    # print(predicted_token_ids)

    # decoded = tokenizer.decode(predicted_token_ids[0].tolist())
    # print(decoded)


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
