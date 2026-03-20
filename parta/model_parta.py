import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List

MAX_SEQ_LEN = 512


class Vocab_Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor):
        embeds = self.vocab_embed(input_ids)
        return embeds


class Vocab_Unembedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, hidden_state: torch.Tensor):
        logits = self.vocab_unembed(hidden_state)
        return logits


class Positional_Encoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.max_len = MAX_SEQ_LEN
        self.d_model = d_model

        position_encodings = torch.zeros((self.max_len, d_model))
        positions = torch.arange(0, self.max_len).unsqueeze(1)

        base = torch.tensor(10000.0)
        two_i = torch.arange(0, d_model, 2)
        base_pow_2i_by_d = torch.exp(-(two_i / d_model) * (torch.log(base)))

        pos_into_exp = positions * base_pow_2i_by_d
        position_encodings[:, 0::2] = torch.sin(pos_into_exp)
        position_encodings[:, 1::2] = torch.cos(pos_into_exp)

        position_encodings = position_encodings.unsqueeze(0)

        self.register_buffer("position_encodings", position_encodings)

    def forward(self, input_ids: torch.Tensor):
        s = input_ids.shape[1]

        if s > self.max_len:
            positional_encodings = torch.zeros(
                (s, self.d_model), device=input_ids.device
            )
            positions = torch.arange(0, s, device=input_ids.device).unsqueeze(1)

            base = torch.tensor(10000.0, device=input_ids.device)
            two_i = torch.arange(0, self.d_model, 2, device=input_ids.device)
            base_pow_2i_by_d = torch.exp(-(two_i / self.d_model) * (torch.log(base)))

            pos_into_exp = positions * base_pow_2i_by_d
            positional_encodings[:, 0::2] = torch.sin(pos_into_exp)
            positional_encodings[:, 1::2] = torch.cos(pos_into_exp)

            positional_encodings = positional_encodings.unsqueeze(0)
            return positional_encodings

        return self.position_encodings[:, :s]


class Feed_Forward_Network(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        hidden_dim = 4 * d_model
        self.up = nn.Linear(d_model, hidden_dim, bias=True)
        self.down = nn.Linear(hidden_dim, d_model, bias=True)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.down(self.gelu(self.up(x)))


class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, mode, tau):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.mode = mode
        self.tau = tau
        self.tanh_clip = mode == "tanh-clipped"

        self.o_mat = nn.Linear(d_model, d_model, bias=False)
        self.qkv_mat = nn.Linear(d_model, 3 * d_model, bias=False)

        causal = torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool))
        self.register_buffer("causal_mask", causal)

    def forward(self, x, attention_mask):
        B, T, _ = x.shape

        q, k, v = self.qkv_mat(x).split(self.d_model, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        s = torch.matmul(q, k.transpose(-2, -1)) / self.d_head ** (1 / 2)

        if self.tanh_clip:
            s = self.tau * torch.tanh(s)

        if T > MAX_SEQ_LEN:
            causal_mask = torch.tril(
                torch.ones(T, T, dtype=torch.bool, device=x.device)
            )
        else:
            causal_mask = self.causal_mask[:T, :T]

        s = s.masked_fill(causal_mask == 0, -torch.inf)

        pad_mask = attention_mask.bool().view(B, 1, 1, T)
        s = s.masked_fill(pad_mask == 0, -torch.inf)

        attention = torch.softmax(s, dim=-1)

        v = torch.matmul(attention, v)
        v = v.transpose(1, 2).reshape(B, T, self.d_model)
        v = self.o_mat(v)
        v = v * pad_mask.squeeze(1).transpose(-1, -2).float()

        return v


class Transformer_Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, mode, tau):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.layer_norm2 = nn.LayerNorm(d_model, elementwise_affine=True)

        self.ffn = Feed_Forward_Network(d_model)
        self.multi_head_attention = Multi_Head_Attention(
            d_model, n_heads, d_head, mode, tau
        )

    def forward(self, x, attention_mask):
        after_attention = self.multi_head_attention(self.layer_norm1(x), attention_mask)
        x = x + after_attention

        after_ffn = self.ffn(self.layer_norm2(x))
        x = x + after_ffn
        return x


class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        super().__init__()

        self.config = config
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.d_head = config["d_head"]
        self.n_layers = config["n_layers"]
        self.vocab_size = config["vocab_size"]
        self.mode = config["mode"]

        if self.mode == "tanh-clipped":
            self.tau = config["tau"]
        else:
            self.tau = None

        self.vocab_embedding = Vocab_Embedding(self.vocab_size, self.d_model)
        self.vocab_unembedding = Vocab_Unembedding(self.vocab_size, self.d_model)

        self.positional_encoding = Positional_Encoding(self.d_model)

        self.transformer_blocks = nn.ModuleList(
            [
                Transformer_Block(
                    self.d_model, self.n_heads, self.d_head, self.mode, self.tau
                )
                for i in range(self.n_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(self.d_model, elementwise_affine=True)

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """

        with torch.no_grad():
            self.vocab_embedding.vocab_embed.weight.copy_(
                weights["W_vocab"].T.contiguous()
            )
            self.vocab_unembedding.vocab_unembed.weight.copy_(
                weights["W_devocab"].T.contiguous()
            )

            self.final_layer_norm.weight.copy_(weights["gamma_final"])
            self.final_layer_norm.bias.copy_(weights["beta_final"])

            for i in range(1, self.n_layers + 1):
                transformer_block = self.transformer_blocks[i - 1]
                transformer_block.layer_norm1.weight.copy_(weights[f"gamma_{i}_1"])
                transformer_block.layer_norm1.bias.copy_(weights[f"beta_{i}_1"])
                transformer_block.layer_norm2.weight.copy_(weights[f"gamma_{i}_2"])
                transformer_block.layer_norm2.bias.copy_(weights[f"beta_{i}_2"])

                transformer_block.ffn.up.weight.copy_(weights[f"W_{i}_up"].T)
                transformer_block.ffn.up.bias.copy_(weights[f"b_{i}_up"])
                transformer_block.ffn.down.weight.copy_(weights[f"W_{i}_down"].T)
                transformer_block.ffn.down.bias.copy_(weights[f"b_{i}_down"])

                all_o_weights = weights[f"W_{i}_O"]
                transformer_block.multi_head_attention.o_mat.weight.copy_(
                    all_o_weights.T
                )

                all_q_weights = torch.cat(
                    [weights[f"W_{i}_Q_{h}"] for h in range(1, self.n_heads + 1)], dim=0
                ).T
                all_k_weights = torch.cat(
                    [weights[f"W_{i}_K_{h}"] for h in range(1, self.n_heads + 1)], dim=0
                ).T
                all_v_weights = torch.cat(
                    [weights[f"W_{i}_V_{h}"] for h in range(1, self.n_heads + 1)], dim=0
                ).T
                all_qkv_weights = torch.cat(
                    [all_q_weights, all_k_weights, all_v_weights], dim=0
                )

                transformer_block.multi_head_attention.qkv_mat.weight.copy_(
                    all_qkv_weights
                )

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

        x = x + self.positional_encoding(input_ids)

        for transfomer_block in self.transformer_blocks:
            x = transfomer_block(x, attention_mask)

        x = self.final_layer_norm(x)

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

    batch_input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=PAD_ID
    )

    batch_attention_mask = pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0
    )

    collated_batch = {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
    }

    return collated_batch
