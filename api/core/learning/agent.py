"""
agent.py — D3QN action selection with topic coherence
"""

import random

import numpy as np
import torch

from .models import DuelingQNetwork


def select_action(
    q_net: DuelingQNetwork,
    state: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
    n_concepts: int | None = None,
    epsilon: float = 0.0,
) -> int:
    """Select action with epsilon-greedy + action masking (concept-agnostic).

    Args:
        n_concepts: number of concepts. If None, inferred from action_mask.
    """
    if n_concepts is None:
        n_concepts = len(action_mask) // 6

    if random.random() < epsilon:
        valid = np.where(action_mask)[0]
        return int(np.random.choice(valid))
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = q_net(state_t, n_concepts).squeeze(0)
        mask_t = torch.BoolTensor(action_mask).to(device)
        q_values[~mask_t] = float("-inf")
        return int(q_values.argmax().item())


def select_topic_coherent_action(
    q_net: DuelingQNetwork,
    state: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
    n_blooms: int = 6,
    n_concepts: int | None = None,
    current_topic: str | None = None,
    steps_in_topic: int = 0,
    topic_to_concept_idxs: dict[str, set[int]] | None = None,
    min_steps_per_topic: int = 3,
    topic_bias: float = 2.0,
) -> int:
    """Select action with topic-coherence bias (concept-agnostic)."""
    if n_concepts is None:
        n_concepts = len(action_mask) // n_blooms

    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = q_net(state_t, n_concepts).squeeze(0).cpu().numpy()

    mask_t = action_mask.astype(bool)
    q_values[~mask_t] = float("-inf")

    if current_topic and steps_in_topic < min_steps_per_topic and topic_to_concept_idxs:
        same_topic_idxs = topic_to_concept_idxs.get(current_topic, set())
        for action_id in range(len(q_values)):
            if not mask_t[action_id]:
                continue
            concept_idx = action_id // n_blooms
            if concept_idx in same_topic_idxs:
                q_values[action_id] += topic_bias

    return int(np.argmax(q_values))


def decode_action(action: int, n_blooms: int = 6):
    """Decode flat action → (concept_idx, bloom_level)."""
    concept_idx = action // n_blooms
    bloom_level = (action % n_blooms) + 1
    return concept_idx, bloom_level
