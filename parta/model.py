import torch
import torch.nn as nn
from typing import Any, Dict


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
    def __init__(self, d_head):
        super().__init__()
        self.d_head = d_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, T, device):
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Feed_Forward_Network(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        hidden_dim = 4 * d_model
        self.up = nn.Linear(d_model, hidden_dim, bias=True)
        self.down = nn.Linear(hidden_dim, d_model, bias=True)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.dropout(self.down(self.ffn_norm(self.gelu(self.up(x)))))


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

        self.rope = Positional_Encoding(d_head)

        self.q_norm = nn.LayerNorm(d_head, elementwise_affine=True)
        self.k_norm = nn.LayerNorm(d_head, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(d_head, elementwise_affine=True)

        self.attention_dropout = nn.Dropout(0.3)

    def forward(self, x, attention_mask):
        B, T, _ = x.shape

        q, k, v = self.qkv_mat(x).split(self.d_model, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        freqs = self.rope(T, x.device)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        q = self.q_norm(q * cos + rotate_half(q) * sin)
        k = self.k_norm(k * cos + rotate_half(k) * sin)
        v = self.v_norm(v)

        s = torch.matmul(q, k.transpose(-2, -1)) / self.d_head ** (1 / 2)

        if self.tanh_clip:
            s = self.tau * torch.tanh(s)

        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))

        s = s.masked_fill(causal_mask == 0, -torch.inf)

        pad_mask = attention_mask.bool().view(B, 1, 1, T)
        s = s.masked_fill(pad_mask == 0, -torch.inf)

        attention = torch.softmax(s, dim=-1)
        attention = self.attention_dropout(attention)

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

        self.transformer_blocks = nn.ModuleList(
            [
                Transformer_Block(
                    self.d_model, self.n_heads, self.d_head, self.mode, self.tau
                )
                for i in range(self.n_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(self.d_model, elementwise_affine=True)
        self.dropout = nn.Dropout(0.3)

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

        x = self.dropout(x)

        for transfomer_block in self.transformer_blocks:
            x = transfomer_block(x, attention_mask)

        x = self.final_layer_norm(x)

        logits = self.vocab_unembedding(x)

        return logits
