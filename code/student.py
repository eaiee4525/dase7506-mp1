from torch import nn
from torch.nn import functional as F
from model import GPT, Block


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()

        # Baseline GELU MLP uses hidden size = 4 * width.
        # SwiGLU needs two input projections, so we use about 8/3 * width
        # to keep the total parameter count almost the same.
        hidden = (8 * width) // 3

        self.value = nn.Linear(width, hidden)
        self.gate = nn.Linear(width, hidden)
        self.out = nn.Linear(hidden, width)
        self.activation = nn.SiLU()

    def forward(self, x):
        value = self.value(x)
        gate = self.activation(self.gate(x))
        return self.out(value * gate)


class SwiGLUBlock(Block):
    def __init__(self, width=128, heads=4, dropout=0.0):
        super().__init__(width, heads)
        self.mlp = SwiGLU(width)
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape

        q, k, v = (
            self.qkv(self.norm1(x))
            .view(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
        )

        attention_output = self.proj(
            attended.transpose(1, 2).reshape(batch, length, width)
        )
        x = x + self.attention_dropout(attention_output)

        mlp_output = self.mlp(self.norm2(x))
        return x + self.mlp_dropout(mlp_output)


class StudentGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        dropout = config.get("dropout", 0.0)

        # Replace the baseline GELU blocks with otherwise identical
        # Transformer blocks using SwiGLU MLPs.
        self.blocks = nn.ModuleList([
            SwiGLUBlock(
                config["width"],
                config["heads"],
                dropout,
            )
            for _ in range(config["depth"])
        ])

        # Match the baseline initialization scheme.
        self.blocks.apply(self.initialize)


def build_model(config):
    return StudentGPT(config)