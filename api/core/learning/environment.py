"""
environment.py — Adaptive Learning Environment for API use.

Mirrors the training environment (saint_rl/env_rl.py): the DQN policy consumes a
concept-agnostic observation whose per-concept block is 24-dimensional
(bloom_mastery 6 + visited 1 + prereq_ok 1 + PCA-reduced concept embedding 16),
and the reward is the dense-only mastery delta anchored at Bloom-3 (Apply).

Serving recomputes the full Bloom-mastery matrix on every observation build so
the state fed to the frozen policy is always consistent (no lazy staleness).
"""

import copy
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
from loguru import logger
import numpy as np
import torch

from .models import SaintModel

_N_BLOOMS = 6
_BASE_FEAT_DIM = 8  # bloom_mastery(6) + visited(1) + prereq_ok(1)
_BLOOM_APPLY_IDX = 2  # index for Bloom level 3 (Apply) in 0-based array
_INITIAL_MASTERY = 0.5
_DEFAULT_PCA_DIM = 16


class AdaptiveLearningEnv(gym.Env):
    """
    Gymnasium environment for adaptive concept sequencing.

    Concept-agnostic observation (global_dim + N * CONCEPT_FEAT_DIM):
      [0 : global_dim]                           global state
        [0 : d_model]                              SAINT encoder hidden state
        [d_model]                                   step / max_steps
        [d_model+1]                                 coverage fraction
      [global_dim : global_dim + N*CONCEPT_FEAT_DIM]  per-concept features (xN)
        Per concept i (8 + pca_dim dims):
          [0:6]   bloom_mastery (6 levels)
          [6]     visited (0/1)
          [7]     prereq_ok (0/1)
          [8:]    PCA-reduced concept embedding

    Action: flat index 0..(N*6-1) -> concept = action // 6, bloom = (action % 6) + 1
    Reward: dense-only mastery delta at Bloom 3.
    """

    metadata: dict[str, list[str]] = {"render_modes": []}  # noqa: RUF012
    N_BLOOMS = _N_BLOOMS
    BASE_FEAT_DIM = _BASE_FEAT_DIM  # bloom_mastery(6) + visited(1) + prereq_ok(1)

    def __init__(
        self,
        saint_model: SaintModel,
        concept_map: dict[str, int],
        prereq_graph: dict[int, list[int]] | None = None,
        concept_blooms: dict[int, list[int]] | None = None,
        max_steps: int = 9999,
        mastery_threshold: float = 0.75,
        dense_coeff: float = 1.0,
        *,
        mask_mastered: bool = True,
        deterministic_train: bool = False,
        max_seq_len: int = 200,
        device: str | None = None,
        external_embeddings: Optional["torch.Tensor"] = None,
        concept_embed_pca: Optional["np.ndarray"] = None,
    ) -> None:
        super().__init__()

        self._model = saint_model
        self._concept_map = concept_map
        self.n_concepts = len(concept_map)
        self._id_to_concept = {v: k for k, v in concept_map.items()}
        # Per-session concept embeddings (None = use model's training embeddings)
        self._external_embeddings = external_embeddings

        self._prereq_graph = prereq_graph or {}
        # Transitive prerequisite closure:
        # concept c is unlocked only when ALL ancestor prerequisites of c are mastered.
        self._prereq_ancestors = self._build_prereq_ancestors()
        # Use caller-provided bloom availability per concept; default to all 6 levels.
        provided_blooms = concept_blooms or {}
        self._concept_blooms: dict[int, list[int]] = {
            i: list(provided_blooms.get(i, range(1, _N_BLOOMS + 1))) for i in range(self.n_concepts)
        }

        self.max_steps = max_steps
        self.mastery_threshold = mastery_threshold
        self.dense_coeff = dense_coeff
        self.mask_mastered = mask_mastered
        self.deterministic_train = deterministic_train
        self.max_seq_len = max_seq_len

        if device is None:
            self.device = next(saint_model.parameters()).device
        else:
            self.device = torch.device(device)

        # PCA-reduced concept embeddings for the per-concept observation block.
        # Shape (n_concepts, pca_dim); zeros when unavailable (keeps obs shape valid).
        self.concept_embed_pca_dim = (
            int(concept_embed_pca.shape[1]) if concept_embed_pca is not None else _DEFAULT_PCA_DIM
        )
        if concept_embed_pca is not None:
            pca = np.asarray(concept_embed_pca, dtype=np.float32)
            if pca.shape != (self.n_concepts, self.concept_embed_pca_dim):
                raise ValueError(
                    f"concept_embed_pca shape {pca.shape} != "
                    f"({self.n_concepts}, {self.concept_embed_pca_dim})"
                )
            self._concept_embed_pca = pca
        else:
            self._concept_embed_pca = np.zeros(
                (self.n_concepts, self.concept_embed_pca_dim), dtype=np.float32
            )

        self.CONCEPT_FEAT_DIM = self.BASE_FEAT_DIM + self.concept_embed_pca_dim

        self.d_model = saint_model.d_model
        self.global_dim = self.d_model + 2  # hidden_state + step_progress + coverage

        obs_dim = self.global_dim + self.n_concepts * self.CONCEPT_FEAT_DIM
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.n_actions = self.n_concepts * self.N_BLOOMS
        self.action_space = spaces.Discrete(self.n_actions)

        self._concept_history: list[int] = []
        self._bloom_history: list[int] = []
        self._response_history: list[int] = []
        self._visited: set = set()
        self._visit_counts: dict[int, int] = {}
        self._step_count: int = 0
        self._current_mastery = np.full(self.n_concepts, _INITIAL_MASTERY, dtype=np.float32)
        self._bloom_mastery = np.full(
            (self.n_concepts, self.N_BLOOMS), _INITIAL_MASTERY, dtype=np.float32
        )
        self._current_hidden = np.zeros(self.d_model, dtype=np.float32)
        self._initial_mastery: float = _INITIAL_MASTERY

        self._precompute_valid_bloom_mask()

    def _precompute_valid_bloom_mask(self) -> Any:
        self._valid_bloom_mask = np.zeros(self.n_actions, dtype=bool)
        for c in range(self.n_concepts):
            valid_blooms = self._concept_blooms.get(c, list(range(1, _N_BLOOMS + 1)))
            for b in valid_blooms:
                if 1 <= b <= _N_BLOOMS:
                    self._valid_bloom_mask[c * self.N_BLOOMS + (b - 1)] = True
            if not self._valid_bloom_mask[c * self.N_BLOOMS : (c + 1) * self.N_BLOOMS].any():
                self._valid_bloom_mask[c * self.N_BLOOMS : (c + 1) * self.N_BLOOMS] = True

    def _decode_action(self, action: int) -> Any:
        concept = action // self.N_BLOOMS
        bloom = (action % self.N_BLOOMS) + 1
        return concept, bloom

    def _build_prereq_ancestors(self) -> dict[int, list[int]]:
        """Precompute transitive prerequisite closure for each concept index."""
        memo: dict[int, set[int]] = {}
        visiting: set[int] = set()

        def dfs(node: int) -> set[int]:
            if node in memo:
                return memo[node]
            if node in visiting:
                # Cycle guard: return direct parents to avoid infinite recursion.
                return {p for p in self._prereq_graph.get(node, []) if 0 <= p < self.n_concepts}

            visiting.add(node)
            ancestors: set[int] = set()
            for parent in self._prereq_graph.get(node, []):
                if not (0 <= parent < self.n_concepts):
                    continue
                ancestors.add(parent)
                ancestors.update(dfs(parent))
            visiting.remove(node)
            memo[node] = ancestors
            return ancestors

        for concept_idx in range(self.n_concepts):
            dfs(concept_idx)
        return {
            concept_idx: sorted(memo.get(concept_idx, set()))
            for concept_idx in range(self.n_concepts)
        }

    def _compute_prereq_ok_mask(self, threshold: float) -> np.ndarray:
        """Compute concept unlock mask from transitive prerequisite closure.

        A concept is unlocked only when every ancestor prerequisite reaches the
        Bloom-3 (Apply) mastery threshold.
        """
        bloom3_mastery = self._bloom_mastery[:, _BLOOM_APPLY_IDX]  # (N,)
        concept_prereq_ok = np.ones(self.n_concepts, dtype=bool)
        for c, ancestors in self._prereq_ancestors.items():
            if ancestors and np.any(bloom3_mastery[ancestors] < threshold):
                concept_prereq_ok[c] = False
        return concept_prereq_ok

    def get_prereq_ok_mask(self, threshold: float | None = None) -> np.ndarray:
        """Public helper for session/graph APIs to read current lock state."""
        th = self.mastery_threshold if threshold is None else float(threshold)
        self._compute_mastery_vector()
        return self._compute_prereq_ok_mask(th).copy()

    def _build_history_tensors(self, seq_len: Any, seq_t: Any, batch_size: Any = 1) -> Any:
        concept_hist = self._concept_history[-seq_t:] if seq_t > 0 else []
        bloom_hist = self._bloom_history[-seq_t:] if seq_t > 0 else []
        response_hist = self._response_history[-seq_t:] if seq_t > 0 else []

        exercise_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
        bloom_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)

        if seq_t > 0:
            c_tensor = torch.tensor(concept_hist, dtype=torch.long, device=self.device) + 1
            b_tensor = torch.tensor(bloom_hist, dtype=torch.long, device=self.device)
            exercise_ids[:, :seq_t] = c_tensor.unsqueeze(0).expand(batch_size, -1)
            bloom_ids[:, :seq_t] = b_tensor.unsqueeze(0).expand(batch_size, -1)

        sos_idx = self._model.SOS_IDX
        decoder_input = torch.full(
            (batch_size, seq_len), sos_idx, dtype=torch.long, device=self.device
        )
        if seq_t > 0:
            r_tensor = torch.tensor(response_hist, dtype=torch.long, device=self.device)
            decoder_input[:, 1 : seq_t + 1] = r_tensor.unsqueeze(0).expand(batch_size, -1)

        return exercise_ids, bloom_ids, decoder_input

    @torch.no_grad()
    def _compute_hidden_state(self) -> Any:
        t_full = len(self._concept_history)
        seq_t = min(t_full, self.max_seq_len - 1)
        seq_len = seq_t + 1

        exercise_ids, bloom_ids, decoder_input = self._build_history_tensors(
            seq_len, seq_t, batch_size=1
        )
        exercise_ids[0, seq_t] = 1
        bloom_ids[0, seq_t] = _BLOOM_APPLY_IDX + 1  # Bloom level 3 (Apply)

        hidden_state, _ = self._model.get_state_and_predictions(
            concept_ids=exercise_ids,
            bloom_levels=bloom_ids,
            decoder_input=decoder_input,
            query_position=seq_t,
            external_embeddings=self._external_embeddings,
        )
        self._current_hidden = hidden_state[0].cpu().numpy()

    @torch.no_grad()
    def _compute_mastery_vector(self) -> Any:
        self._compute_bloom_mastery_vector()
        # Concept mastery = Bloom 3 (Apply) mastery.
        # Student can still practice higher blooms, but unlock is based on B3.
        self._current_mastery[:] = self._bloom_mastery[:, _BLOOM_APPLY_IDX]

    @torch.no_grad()
    def _compute_bloom_mastery_vector(self) -> Any:
        n_concepts = self.n_concepts
        t_full = len(self._concept_history)
        seq_t = min(t_full, self.max_seq_len - 1)
        seq_len = seq_t + 1

        pairs = [
            (c, b)
            for c in range(n_concepts)
            for b in self._concept_blooms.get(c, list(range(1, _N_BLOOMS + 1)))
            if 1 <= b <= _N_BLOOMS
        ]

        if not pairs:
            return

        chunk_size = 100
        for start in range(0, len(pairs), chunk_size):
            chunk = pairs[start : start + chunk_size]
            n_chunk = len(chunk)

            exercise_ids, bloom_ids, decoder_input = self._build_history_tensors(
                seq_len, seq_t, batch_size=n_chunk
            )

            for i, (c, b) in enumerate(chunk):
                exercise_ids[i, seq_t] = c + 1
                bloom_ids[i, seq_t] = b

            output = self._model(
                concept_ids=exercise_ids,
                bloom_levels=bloom_ids,
                responses=None,
                decoder_input=decoder_input,
                external_embeddings=self._external_embeddings,
            )
            preds = output[:, seq_t].cpu().numpy().astype(np.float32)

            for i, (c, b) in enumerate(chunk):
                self._bloom_mastery[c, b - 1] = preds[i]

    @torch.no_grad()
    def _predict_single_concept(self, concept_idx: Any, bloom_level: Any = None) -> Any:
        t_full = len(self._concept_history)
        seq_t = min(t_full, self.max_seq_len - 1)
        seq_len = seq_t + 1

        exercise_ids, bloom_ids, decoder_input = self._build_history_tensors(
            seq_len, seq_t, batch_size=1
        )
        exercise_ids[0, seq_t] = concept_idx + 1
        bloom_ids[0, seq_t] = bloom_level if bloom_level is not None else (_BLOOM_APPLY_IDX + 1)

        output = self._model(
            concept_ids=exercise_ids,
            bloom_levels=bloom_ids,
            responses=None,
            decoder_input=decoder_input,
            external_embeddings=self._external_embeddings,
        )
        return float(output[0, seq_t].cpu().item())

    def _build_obs(self) -> np.ndarray:
        """Build concept-agnostic observation: global_state + per-concept features.

        Recomputes the full Bloom-mastery matrix first so the observation fed to
        the frozen policy is always consistent with the current interaction history.
        """
        self._compute_mastery_vector()

        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # Global state: [hidden_state(d_model) | step_progress | coverage]
        obs[: self.d_model] = self._current_hidden
        obs[self.d_model] = self._step_count / max(self.max_steps, 1)
        obs[self.d_model + 1] = len(self._visited) / self.n_concepts

        # Per-concept features: [bloom_mastery(6) | visited(1) | prereq_ok(1) | PCA] x N
        concept_prereq_ok = self._compute_prereq_ok_mask(self.mastery_threshold).astype(np.float32)

        visited_arr = np.zeros(self.n_concepts, dtype=np.float32)
        for c in self._visited:
            visited_arr[c] = 1.0

        per_concept = np.empty((self.n_concepts, self.CONCEPT_FEAT_DIM), dtype=np.float32)
        per_concept[:, :_N_BLOOMS] = self._bloom_mastery  # (N, 6)
        per_concept[:, 6] = visited_arr  # visited
        per_concept[:, 7] = concept_prereq_ok  # prereq_ok
        per_concept[:, 8:] = self._concept_embed_pca  # PCA embedding

        obs[self.global_dim :] = per_concept.ravel()
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, int | float]]:
        del options
        super().reset(seed=seed)
        self._concept_history = []
        self._bloom_history = []
        self._response_history = []
        self._visited = set()
        self._visit_counts = {}
        self._step_count = 0
        self._bloom_mastery = np.full(
            (self.n_concepts, self.N_BLOOMS), _INITIAL_MASTERY, dtype=np.float32
        )

        self._compute_hidden_state()
        self._compute_mastery_vector()
        self._initial_mastery = float(np.mean(self._current_mastery))

        return self._build_obs(), {"step": 0, "avg_mastery": self._initial_mastery}

    def inject_history(
        self,
        concept_indices: list,
        bloom_levels: list,
        responses: list,
    ) -> Any:
        """Replay saved subject progress into the environment.

        This feeds concept/bloom/response triples from persisted subject-level
        history so SAINT can make predictions based on the full learning context.
        After injection, mastery and hidden state reflect the restored subject progress.
        """
        assert len(concept_indices) == len(bloom_levels) == len(responses), (
            f"History length mismatch: {len(concept_indices)} concepts, {len(bloom_levels)} blooms, {len(responses)} responses"
        )

        for c_idx, bloom, resp in zip(concept_indices, bloom_levels, responses, strict=False):
            self._concept_history.append(c_idx)
            self._bloom_history.append(bloom)
            self._response_history.append(int(resp))
            self._visited.add(c_idx)
            self._visit_counts[c_idx] = self._visit_counts.get(c_idx, 0) + 1

        self._step_count = len(concept_indices)

        # Recompute SAINT hidden state and mastery from full history
        if len(concept_indices) > 0:
            self._compute_hidden_state()
            self._compute_mastery_vector()

        logger.info(
            "[Env] Replayed {} subject-history interactions, visited {} concepts, avg mastery: {:.4f}",
            len(concept_indices),
            len(self._visited),
            float(np.mean(self._current_mastery)),
        )

    def step(self, action: int, *, human_correct: bool | None = None) -> Any:
        assert self.action_space.contains(action)

        concept, bloom = self._decode_action(action)
        p_before = float(np.clip(self._bloom_mastery[concept, bloom - 1], 1e-6, 1 - 1e-6))
        prev_m_concept = float(self._current_mastery[concept])
        self._visited.add(concept)
        self._visit_counts[concept] = self._visit_counts.get(concept, 0) + 1

        if human_correct is not None:
            r = int(human_correct)
        elif self.deterministic_train:
            r = 1 if p_before >= _INITIAL_MASTERY else 0
        else:
            r = int(self.np_random.binomial(1, p_before))

        self._concept_history.append(int(concept))
        self._bloom_history.append(bloom)
        self._response_history.append(r)
        self._step_count += 1

        self._compute_hidden_state()
        terminated = self._step_count >= self.max_steps

        # Update the played bloom cell; concept mastery is anchored at Bloom-3.
        apply_bloom_level = _BLOOM_APPLY_IDX + 1  # Bloom-3 (Apply)
        p_after = self._predict_single_concept(concept, bloom_level=bloom)
        self._bloom_mastery[concept, bloom - 1] = p_after
        if bloom != apply_bloom_level:
            p_after_b3 = self._predict_single_concept(concept, bloom_level=apply_bloom_level)
            self._bloom_mastery[concept, _BLOOM_APPLY_IDX] = p_after_b3
            self._current_mastery[concept] = p_after_b3
        else:
            self._current_mastery[concept] = p_after

        # Dense-only reward: mastery delta at Bloom-3 (Apply).
        mastery_delta = float(self._current_mastery[concept]) - prev_m_concept
        reward = self.dense_coeff * mastery_delta

        info = {
            "step": self._step_count,
            "concept": self._id_to_concept.get(concept, str(concept)),
            "concept_idx": concept,
            "bloom": bloom,
            "p_correct": p_before,
            "mastery_delta": float(mastery_delta),
            "response": r,
            "reward": float(reward),
            "avg_mastery": float(np.mean(self._current_mastery)),
        }
        return self._build_obs(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Return the valid action mask: prerequisite, sequential Bloom, mastery cap."""
        self._compute_mastery_vector()
        th = self.mastery_threshold
        n, b = self.n_concepts, self.N_BLOOMS
        bm = self._bloom_mastery  # (N, 6)

        prereq_ok_2d = np.repeat(
            self._compute_prereq_ok_mask(th)[:, np.newaxis], b, axis=1
        )  # (N, 6)
        valid_2d = self._valid_bloom_mask.reshape(n, b)  # (N, 6)

        # Sequential Bloom unlock: Bloom k requires Bloom k-1 mastery >= th.
        seq_ok = np.ones((n, b), dtype=bool)
        seq_ok[:, 1:] = bm[:, :-1] >= th

        # Mastery cap: skip already-mastered concept-Bloom pairs.
        not_mastered = np.ones((n, b), dtype=bool)
        if self.mask_mastered:
            not_mastered = bm < th

        masks = (prereq_ok_2d & valid_2d & seq_ok & not_mastered).ravel()

        if not masks.any():
            # Fallback 1: keep prerequisite lock, only relax "mask_mastered".
            masks = (prereq_ok_2d & valid_2d & seq_ok).ravel()

        if not masks.any():
            # Fallback 2 (degenerate graph/cycle): keep env usable instead of crashing.
            masks = self._valid_bloom_mask.copy()
        result: np.ndarray = masks
        return result

    def get_mastery_matrix(self) -> np.ndarray:
        """Return bloom mastery matrix (n_concepts, 6)."""
        self._compute_mastery_vector()
        return self._bloom_mastery.copy()

    def get_concept_mastery(self) -> np.ndarray:
        """Return per-concept mastery vector."""
        self._compute_mastery_vector()
        return self._current_mastery.copy()

    def get_session_stats(self) -> dict[str, int | float | dict[int, int]]:
        """Return current session statistics."""
        return {
            "step": self._step_count,
            "max_steps": self.max_steps,
            "concepts_visited": len(self._visited),
            "total_concepts": self.n_concepts,
            "avg_mastery": float(np.mean(self._current_mastery)),
            "coverage": len(self._visited) / self.n_concepts,
            "visit_counts": dict(self._visit_counts),
        }

    def is_concept_visited(self, concept_idx: int) -> bool:
        """Return whether a concept has been visited in this session."""
        return concept_idx in self._visited

    def get_visit_count(self, concept_idx: int) -> int:
        """Return number of interactions for a concept in this session."""
        return int(self._visit_counts.get(concept_idx, 0))

    def create_snapshot(self) -> "AdaptiveLearningEnv":
        """Create a lightweight copy for simulation.

        Shares read-only references (model, embeddings, graphs) but deep-copies
        all mutable state so that calling ``step()`` on the snapshot does not
        affect the original environment.
        """
        snap = copy.copy(self)  # shallow — shares _model, _external_embeddings, etc.
        # Deep-copy every mutable field
        snap._concept_history = list(self._concept_history)
        snap._bloom_history = list(self._bloom_history)
        snap._response_history = list(self._response_history)
        snap._visited = set(self._visited)
        snap._visit_counts = dict(self._visit_counts)
        snap._bloom_mastery = self._bloom_mastery.copy()
        snap._current_mastery = self._current_mastery.copy()
        snap._current_hidden = self._current_hidden.copy()
        # _concept_embed_pca is read-only; shallow copy shares it safely.
        return snap
