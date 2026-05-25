"""PrerequisiteClassifier — MLP for prerequisite link prediction.

Architecture: [emb_A || emb_B] (1536-d) -> 512 -> 256 -> 1 logit.

Mirrors the network used in module1_update training (LectureBank, F1=0.825,
AUC=0.908). Output is a raw logit; apply torch.sigmoid for probability.
"""

import torch
from torch import Tensor, nn


class PrerequisiteClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        input_dim = 2 * embedding_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, emb_a: Tensor, emb_b: Tensor) -> Tensor:
        x = torch.cat([emb_a, emb_b], dim=-1)
        out: Tensor = self.net(x)
        return out
