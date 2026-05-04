"""
models.py — PyTorch model classes (SAINT + DuelingQNetwork)
Ported from saint_model.py and train_dqn.py
"""

import math

import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SaintModel(nn.Module):
    """SAINT Knowledge Tracing Model with Bloom Taxonomy."""

    SOS_IDX = 3

    def __init__(
        self,
        concept_embeddings,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        d_ff: int = 512,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        bloom_levels: int = 7,
        bloom_emb_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.bloom_levels = bloom_levels

        if isinstance(concept_embeddings, np.ndarray):
            concept_embeddings = torch.from_numpy(concept_embeddings).float()

        n_concepts, emb_dim = concept_embeddings.shape
        padded = torch.zeros(n_concepts + 1, emb_dim)
        padded[1:] = concept_embeddings
        self.register_buffer("concept_emb_matrix", padded)
        self.n_concepts = n_concepts

        self.concept_proj = nn.Linear(emb_dim, d_model)
        self.bloom_emb = nn.Embedding(bloom_levels, bloom_emb_dim, padding_idx=0)
        self.bloom_proj = nn.Linear(bloom_emb_dim, d_model)
        self.response_emb = nn.Embedding(4, d_model, padding_idx=2)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 2, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        self.output_proj = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _responses_to_idx(self, responses: torch.Tensor) -> torch.Tensor:
        idx = responses.long().clamp(0, 1)
        idx[responses < 0] = 2
        return idx

    def forward(
        self,
        concept_ids: torch.Tensor,
        bloom_levels: torch.Tensor,
        responses: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        decoder_input: torch.Tensor | None = None,
        external_embeddings: torch.Tensor | None = None,
        *,
        return_logits: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len = concept_ids.shape
        device = concept_ids.device

        emb_source = (
            external_embeddings if external_embeddings is not None else self.concept_emb_matrix
        )
        concept_emb = emb_source[concept_ids]
        concept_out = self.concept_proj(concept_emb)
        bloom_emb = self.bloom_emb(bloom_levels)
        bloom_out = self.bloom_proj(bloom_emb)
        encoder_input = concept_out + bloom_out
        encoder_input = self.pos_enc(encoder_input)

        if src_key_padding_mask is None:
            src_key_padding_mask = concept_ids == 0

        memory = self.encoder(encoder_input, src_key_padding_mask=src_key_padding_mask)

        if decoder_input is not None:
            dec_emb = self.response_emb(decoder_input)
        else:
            resp_idx = self._responses_to_idx(responses)
            shifted = torch.full(
                (batch_size, seq_len), self.SOS_IDX, dtype=torch.long, device=device
            )
            shifted[:, 1:] = resp_idx[:, :-1]
            dec_emb = self.response_emb(shifted)

        dec_emb = dec_emb + bloom_out
        dec_emb = self.pos_enc(dec_emb)
        causal_mask = self._make_causal_mask(seq_len, device)

        dec_out = self.decoder(
            dec_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.output_proj(dec_out).squeeze(-1)

        if return_logits:
            return logits
        return torch.sigmoid(logits)

    @torch.no_grad()
    def get_state_and_predictions(
        self,
        concept_ids: torch.Tensor,
        bloom_levels: torch.Tensor,
        decoder_input: torch.Tensor,
        query_position: int,
        external_embeddings: torch.Tensor | None = None,
    ):
        _batch_size, seq_len = concept_ids.shape
        device = concept_ids.device

        emb_source = (
            external_embeddings if external_embeddings is not None else self.concept_emb_matrix
        )
        concept_emb = emb_source[concept_ids]
        concept_out = self.concept_proj(concept_emb)
        bloom_emb = self.bloom_emb(bloom_levels)
        bloom_out = self.bloom_proj(bloom_emb)
        encoder_input = concept_out + bloom_out
        encoder_input = self.pos_enc(encoder_input)
        src_key_padding_mask = concept_ids == 0
        memory = self.encoder(encoder_input, src_key_padding_mask=src_key_padding_mask)

        dec_emb = self.response_emb(decoder_input)
        dec_emb = dec_emb + bloom_out
        dec_emb = self.pos_enc(dec_emb)
        causal_mask = self._make_causal_mask(seq_len, device)
        dec_out = self.decoder(
            dec_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.output_proj(dec_out).squeeze(-1)
        predictions = torch.sigmoid(logits[:, query_position])

        mask = ~src_key_padding_mask
        mask_expanded = mask.unsqueeze(-1).float()
        hidden_state = (memory * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return hidden_state, predictions


class DuelingQNetwork(nn.Module):
    """Concept-agnostic Dueling DQN with per-concept scoring.

    Architecture:
        Per concept i, input = [global_state | concept_feat_i] → shared backbone → Dueling V(1)+A(6)
        Weights are independent of N (number of concepts). Works with any N at inference.

    Observation layout (flat):
        [0 : global_dim]                          global state (SAINT hidden + progress)
        [global_dim : global_dim + N*concept_feat_dim]  per-concept features (bloom*6 + visited + prereq_ok)
    """

    CONCEPT_FEAT_DIM = 8  # bloom_mastery(6) + visited(1) + prereq_ok(1)
    N_BLOOMS = 6

    def __init__(self, global_dim=130, hidden_sizes=(256, 256)):
        super().__init__()
        self.global_dim = global_dim
        input_dim = global_dim + self.CONCEPT_FEAT_DIM  # e.g. 138

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.backbone = nn.Sequential(*layers)

        self.value = nn.Sequential(nn.Linear(prev, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(
            nn.Linear(prev, 128), nn.ReLU(), nn.Linear(128, self.N_BLOOMS)
        )

    def forward(self, flat_obs, n_concepts):
        """
        Args:
            flat_obs:    (B, global_dim + n_concepts * CONCEPT_FEAT_DIM)
            n_concepts:  int — number of concepts (can vary between calls)
        Returns:
            Q values:    (B, n_concepts * N_BLOOMS)
        """
        batch_size = flat_obs.shape[0]
        global_state = flat_obs[:, : self.global_dim]
        concept_flat = flat_obs[:, self.global_dim :]
        concept_features = concept_flat.view(batch_size, n_concepts, self.CONCEPT_FEAT_DIM)

        g = global_state.unsqueeze(1).expand(batch_size, n_concepts, -1)
        x = torch.cat([g, concept_features], dim=-1)
        x = x.view(batch_size * n_concepts, -1)

        features = self.backbone(x)
        v = self.value(features)
        a = self.advantage(features)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q.view(batch_size, n_concepts * self.N_BLOOMS)


def load_saint_model(checkpoint_path: str, device: torch.device):
    """Load SAINT model from checkpoint. Returns (model, concept_map, config)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    concept_map = checkpoint.get("concept_map", {})

    # Extract concept embeddings from checkpoint
    state_dict = checkpoint["model_state_dict"]
    concept_emb_matrix = state_dict["concept_emb_matrix"]  # (N+1, 768)
    embeddings = concept_emb_matrix[1:].cpu().numpy()  # Remove PAD row

    model = SaintModel(
        concept_embeddings=embeddings,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_encoder_layers=cfg["n_encoder_layers"],
        n_decoder_layers=cfg["n_decoder_layers"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"] + 1,
        dropout=0.0,
        bloom_levels=cfg.get("bloom_levels", 7),
        bloom_emb_dim=cfg.get("bloom_emb_dim", 32),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, concept_map, cfg


def load_dqn_model(checkpoint_path: str, device: torch.device):
    """Load concept-agnostic DQN from checkpoint. Returns (model, config_info)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    global_dim = ckpt.get("global_dim", 130)
    hidden_sizes = ckpt.get("hidden_sizes", (256, 256))

    q_net = DuelingQNetwork(global_dim=global_dim, hidden_sizes=hidden_sizes).to(device)
    q_net.load_state_dict(ckpt["q_net_state_dict"])
    q_net.eval()

    return q_net, {
        "global_dim": global_dim,
        "hidden_sizes": hidden_sizes,
        "step": ckpt.get("step"),
        "mean_reward": ckpt.get("mean_reward"),
    }
