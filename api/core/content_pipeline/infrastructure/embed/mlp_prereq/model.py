"""PrerequisiteClassifier — MLP for prerequisite link prediction.

Architecture: pair feature vector -> 512 -> 256 -> 1 logit.

The current production checkpoint is trained on ViMath with BGE-M3 embeddings
and rich pair features: [emb_A, emb_B, |emb_A-emb_B|, emb_A*emb_B]. Output is a
raw logit; apply torch.sigmoid for probability.
"""

from torch import Tensor, nn


class PrerequisiteClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

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

    def forward(self, features: Tensor) -> Tensor:
        out: Tensor = self.net(features)
        return out
