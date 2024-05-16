import torch
import torch.nn as nn
import torch.nn.functional as F
from rapgpt.config import Config


class Head(nn.Module):
    def __init__(
        self, hidden_dim: int, head_size: int, context_length: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.key = nn.Linear(hidden_dim, head_size, bias=False)
        self.query = nn.Linear(hidden_dim, head_size, bias=False)
        self.value = nn.Linear(hidden_dim, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_length, context_length))
        )  # Prevents optimization of tensor
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(
        self,
        hidden_dim: int,
        head_size: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    hidden_dim=hidden_dim,
                    head_size=head_size,
                    context_length=context_length,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.1,
    ) -> None:
        # hidden_dim: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = hidden_dim // num_heads
        self.sa = MultiHeadAttention(
            hidden_dim=hidden_dim,
            head_size=head_size,
            context_length=context_length,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffwd = FeedFoward(hidden_dim, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, config: Config) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.token_embedding_table = nn.Embedding(
            self.vocab_size, self.config.model.hidden_dim
        )
        self.position_embedding_table = nn.Embedding(
            self.config.dataset_encoding.context_length, self.config.model.hidden_dim
        )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    hidden_dim=self.config.model.hidden_dim,
                    num_heads=self.config.model.num_heads,
                    context_length=self.config.dataset_encoding.context_length,
                    dropout=self.config.model.dropout,
                )
                for _ in range(self.config.model.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(self.config.model.hidden_dim)  # final layer norm
        self.lm_head = nn.Linear(self.config.model.hidden_dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """B, T = x.shape"""

        tok_emb = self.token_embedding_table(x)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(x.shape[1], device="cpu")
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        x = self.transformer_blocks(x)  # (B,T,C)

        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        return logits

    @torch.no_grad()
    def generate(self, x: torch.Tensor, new_tokens: int) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(new_tokens):
            # crop x to the last block_size tokens
            idx_cond = x[:, -self.config.dataset_encoding.context_length :]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            x_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            x = torch.cat((x, x_next), dim=1)  # (B, T+1)
        return x