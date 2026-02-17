import torch
import torch.nn as nn
from typing import Any, Dict, List


class Vocab_Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, d_model)

    def set_weights(self, vocab_embed_weights: torch.Tensor):
        self.vocab_embed.weight.copy_(vocab_embed_weights.T.contiguous())

    def forward(self, input_ids: torch.Tensor):
        embeds = self.vocab_embed(input_ids)
        return embeds


class Vocab_Unembedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_unembed = nn.Linear(d_model, vocab_size, bias=False)

    def set_weights(self, vocab_unembed_weights: torch.Tensor):
        self.vocab_unembed.weight.copy_(vocab_unembed_weights.T.contiguous())

    def forward(self, hidden_state: torch.Tensor):
        logits = self.vocab_unembed(hidden_state)
        return logits


class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        self.config = config
        super().__init__()

        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.d_head = config["d_head"]
        self.n_layers = config["n_layers"]
        self.vocab_size = config["vocab_size"]
        self.mode = config["mode"]

        if self.mode == "tanh-clipped":
            self.tau = config["tau"]

        self.vocab_embedding = Vocab_Embedding(self.vocab_size, self.d_model)
        self.vocab_unembedding = Vocab_Unembedding(self.vocab_size, self.d_model)

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """

        with torch.no_grad():
            self.vocab_embedding.set_weights(weights["W_vocab"])
            self.vocab_unembedding.set_weights(weights["W_devocab"])

        # print(weights.keys())
        # for k in weights.keys():
        #     print(k + ": ", weights[k].shape)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Implement the forward pass of the model. The output should be a tensor of shape (T, |Vocab|).

        Parameters:
            - input_ids: A tensor of shape (batch_size, sequence_len) containing token IDs.
            - attention_mask: A tensor of shape (batch_size, sequence_len) containing 1s for valid tokens and 0s for padding.

        Returns:
            - A tensor of shape (batch_size, sequence_len, vocab_size) containing the logits for each token in the vocabulary.
            Logits are the raw, unnormalized scores output by the model, which can be converted to probabilities using a softmax function.
        """

        x = self.vocab_embedding(input_ids)
        logits = self.vocab_unembedding(x)

        return logits


def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function inputs config and weights and outputs a nn.Module derived object.
    """

    model = LanguageModel(config)
    model.set_weights(weights)

    return model


def collate_fn(batch: Dict[str, List[torch.tensor]]) -> Dict[str, torch.Tensor]:
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function takes in a batch of data and outputs a dictionary of tensors ready to be fed into the model.
    """
    PAD_ID = 0  # Assume 0 is the padding token ID

    input_ids_list = batch["input_ids"]
    attention_mask_list = batch["attention_mask"]
    max_seq_len = max(input_ids.shape[0] for input_ids in input_ids_list)

    batch_input_ids = []
    batch_attention_mask = []

    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        pad_len = max_seq_len - input_ids.shape[0]

        padded_input_ids = torch.cat(
            [
                input_ids,
                torch.full((pad_len,), PAD_ID, dtype=input_ids.dtype),
            ]
        )

        padded_attention_mask = torch.cat(
            [
                attention_mask,
                torch.zeros(pad_len, dtype=attention_mask.dtype),
            ]
        )

        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(padded_attention_mask)

    collated_batch = {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
    }

    return collated_batch
