from typing import Any

"""
agent.py — DQN action selection with topic coherence
"""

import secrets

import numpy as np
import torch

from .models import VanillaQNetwork

# SystemRandom instance used for epsilon-greedy exploration (non-cryptographic).
_rng = secrets.SystemRandom()

# Number of Bloom's taxonomy levels (used for action-space indexing).
_N_BLOOMS = 6


def select_action(
    q_net: VanillaQNetwork,
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
        n_concepts = len(action_mask) // _N_BLOOMS

    if _rng.random() < epsilon:
        valid = np.where(action_mask)[0]
        rng_np = np.random.default_rng()
        return int(rng_np.choice(valid))
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = q_net(state_t, n_concepts).squeeze(0)
        mask_t = torch.BoolTensor(action_mask).to(device)
        q_values[~mask_t] = float("-inf")
        return int(q_values.argmax().item())


def decode_action(action: int, n_blooms: int = 6) -> Any:
    """Decode flat action → (concept_idx, bloom_level)."""
    concept_idx = action // n_blooms
    bloom_level = (action % n_blooms) + 1
    return concept_idx, bloom_level


def select_next_concept_action(
    env: Any,
    q_net: VanillaQNetwork,
    state: np.ndarray,
    device: torch.device,
) -> tuple[int, int, int]:
    """Pick the next (concept_idx, bloom_level, action_id) for an env step.

    Step 0 forces a deterministic warm-up (concept 0, Bloom 1); every later step
    masks invalid actions and lets the DQN greedily choose. Shared by the live
    and eager/prefetch exercise paths so both stay in lock-step.
    """
    current_step = env.get_session_stats().get("step", 0)
    if current_step == 0:
        concept_idx, bloom_level = 0, 1
        return concept_idx, bloom_level, concept_idx * _N_BLOOMS + (bloom_level - 1)

    action_id = select_action(
        q_net,
        state,
        env.action_masks(),
        device,
        n_concepts=env.n_concepts,
    )
    concept_idx, bloom_level = decode_action(action_id)
    return concept_idx, bloom_level, action_id
