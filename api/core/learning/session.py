"""
session.py — Session state management (SessionManager + SessionState).

SessionManager handles:
  - Model loading (SAINT + DQN)
  - Session lifecycle (create, get, status)
  - Knowledge graph and mastery queries

Exercise-related logic (generation, submission, prefetch) is delegated
to the learning exercise service.
"""

import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, cast
import uuid

from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from api.config import get_settings
from api.core.shared.persistence import (
    load_pipeline_job_for_user,
    load_subject_progress_by_session_for_user,
    load_subject_progress_for_user,
    save_subject_progress_snapshot,
)

from .bloom import BLOOM_LABEL_SEQUENCE
from .environment import AdaptiveLearningEnv
from .exercise_types.payloads import ExercisePayload
from .models import VanillaQNetwork, load_dqn_model, load_saint_model
from .pca import apply_concept_pca
from .progress_metrics import build_prereq_graph_from_edges, summarize_mastery_progress
from .subject_progress_snapshot import build_subject_progress_snapshot

_MASTERY_THRESHOLD = float(get_settings().adaptive_mastery_threshold)
_MIN_STORED_MAX_STEPS = 50  # sessions stored with max_steps <= this are treated as unbounded
_DEFAULT_MAX_STEPS = 9999


@dataclass
class ExerciseRecord:
    exercise_id: str
    concept_idx: int
    concept_name: str
    bloom_level: int
    question: str
    payload: ExercisePayload
    explanation: str = ""
    explanation_correct: str = ""
    explanation_incorrect: str = ""
    theory: dict[str, str | list[str]] | None = None
    user_answer: str | None = None
    is_correct: bool | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionState:
    session_id: str
    env: AdaptiveLearningEnv
    q_net: VanillaQNetwork
    device: torch.device
    concept_map: dict[str, int]
    concept_names: dict[str, str]
    concept_definitions: dict[str, str]
    prereq_graph: dict[int, list[int]]
    current_obs: np.ndarray
    user_id: str | None = None
    current_exercise: ExerciseRecord | None = None
    exercise_history: list[ExerciseRecord] = field(default_factory=list)
    total_correct: int = 0
    total_answered: int = 0
    job_id: str | None = None
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    status: str = "active"
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    concept_theories: dict[str, dict[str, str | list[str]]] = field(default_factory=dict)
    _prefetch_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    tutor_chat_history: list[dict[str, str]] = field(default_factory=list)
    tutor_chat_exercise_id: str | None = None

    @staticmethod
    def _restore_exercise_records(session: "SessionState", prev_history: list[dict]) -> None:
        from pydantic import TypeAdapter

        adapter: TypeAdapter[ExercisePayload] = TypeAdapter(ExercisePayload)
        for ex in prev_history:
            session.exercise_history.append(
                ExerciseRecord(
                    exercise_id=ex.get("exercise_id", ""),
                    concept_idx=ex["concept_idx"],
                    concept_name=ex.get("concept_name", ""),
                    bloom_level=ex["bloom_level"],
                    question=ex.get("question", ""),
                    payload=adapter.validate_python(ex["payload"]),
                    explanation=ex.get("explanation", ""),
                    explanation_correct=ex.get("explanation_correct", ""),
                    explanation_incorrect=ex.get("explanation_incorrect", ""),
                    theory=ex.get("theory"),
                    user_answer=ex.get("user_answer"),
                    is_correct=ex.get("is_correct"),
                    timestamp=ex.get("timestamp", 0),
                )
            )


class SessionManager:
    """Session lifecycle management and knowledge graph queries.

    Responsibilities:
      - Load SAINT + DQN models once.
      - Create / retrieve sessions.
      - Query session status, knowledge graph, mastery matrix, concept details.

    Exercise flow (generation, submission, prefetch) is handled by
    ExerciseService, which receives SessionState objects from this manager.
    """

    # Sentence-transformer for encoding new concepts (lazy-loaded)
    _text_encoder = None
    _TEXT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

    def __init__(
        self,
        saint_path: str,
        dqn_path: str,
        concepts_data: dict[str, dict[str, object]] | None = None,
        prereq_data: list[dict[str, str]] | None = None,
        device: str | None = None,
    ) -> None:
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._saint_path = saint_path
        self._dqn_path = dqn_path

        # Load models once
        self._saint_model, self._concept_map, self._saint_config = load_saint_model(
            saint_path, self._device
        )
        self._q_net, self._dqn_info = load_dqn_model(dqn_path, self._device)
        self._mastery_threshold = _MASTERY_THRESHOLD

        self._concepts_data = concepts_data or {}
        self._prereq_data = prereq_data or []
        self._prereq_graph = self._build_prereq_graph()
        self._concept_names, self._concept_defs = self._build_concept_info()

        # Active sessions
        self._sessions: dict[str, SessionState] = {}
        self._recovery_locks: dict[str, asyncio.Lock] = {}
        self._subject_session_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._subject_session_ids: dict[tuple[str, str], str] = {}

    # ── Properties ──────────────────────────────────────────

    @property
    def concept_map(self) -> Any:
        return self._concept_map

    @property
    def concept_names(self) -> Any:
        return self._concept_names

    @property
    def n_concepts(self) -> Any:
        return len(self._concept_map)

    # ── Internal helpers ────────────────────────────────────

    def _build_prereq_graph(self) -> dict[int, list[int]]:
        return self._build_prereq_graph_from_edges(self._prereq_data, self._concept_map)

    @staticmethod
    def _build_id_to_concept_map(concept_map: dict[str, int]) -> dict[int, str]:
        return {v: k for k, v in concept_map.items()}

    @staticmethod
    def _build_prereq_graph_from_edges(
        prereq_edges: list[dict],
        concept_map: dict[str, int],
    ) -> dict[int, list[int]]:
        graph = build_prereq_graph_from_edges(prereq_edges, concept_map)
        kept = sum(len(v) for v in graph.values())
        dropped = len(prereq_edges) - kept
        if dropped:
            logger.warning(
                "[Session] _build_prereq_graph_from_edges: dropped {}/{} edges",
                dropped,
                len(prereq_edges),
            )
        return graph

    def _build_concept_info(self) -> Any:
        names, defs, _ = self._build_concept_info_from_data(self._concepts_data, self._concept_map)
        return names, defs

    @classmethod
    def _build_concept_info_from_data(
        cls,
        concepts_data: dict[str, Any],
        concept_map: dict[str, int],
    ) -> Any:
        names: dict[str, str] = {}
        defs: dict[str, str] = {}
        theories: dict[str, dict[str, Any]] = {}
        id_to_concept = cls._build_id_to_concept_map(concept_map)

        for cid_raw, cdata in concepts_data.items():
            cid = str(cid_raw)
            if cid not in concept_map:
                continue
            name = cdata.get("name", cid)
            if not name or str(name).lower() == "nan":
                name = cid
            names[cid] = name
            defs[cid] = cdata.get("definition", "")
            if "theory" in cdata:
                theories[cid] = cdata["theory"]

        for idx in range(len(concept_map)):
            cid = id_to_concept.get(idx, str(idx))
            if cid not in names:
                names[cid] = cid
                defs[cid] = ""

        return names, defs, theories

    @classmethod
    def _get_text_encoder(cls) -> SentenceTransformer:
        """Lazy-load sentence-transformer (same model used in SAINT training)."""
        if cls._text_encoder is None:
            logger.info("[Session] Loading text encoder: {}", cls._TEXT_MODEL_NAME)
            cls._text_encoder = SentenceTransformer(cls._TEXT_MODEL_NAME)
            logger.info("[Session] ✓ Text encoder ready")
        return cast("SentenceTransformer", cls._text_encoder)

    def _encode_concepts(self, concept_names_ordered: list[str]) -> np.ndarray:
        """Encode concept names into 768d embeddings."""
        encoder = self._get_text_encoder()
        embeddings = encoder.encode(concept_names_ordered, show_progress_bar=False, batch_size=32)
        return np.array(embeddings, dtype=np.float32)

    # ── Session Lifecycle ───────────────────────────────────

    def _clean_expired_sessions(self, max_size: Any = 500) -> Any:
        """Simple eviction logic: if we exceed max_size, evict oldest 20% by access time."""
        if len(self._sessions) > max_size:
            sorted_keys = sorted(
                self._sessions.keys(),
                key=lambda k: getattr(
                    self._sessions[k], "accessed_at", getattr(self._sessions[k], "created_at", 0)
                ),
            )
            # Remove oldest 20%
            for k in sorted_keys[: max_size // 5]:
                session = self._sessions.get(k)
                self.remove_session(k)
                if session and session.user_id and session.job_id:
                    subject_key = (session.user_id, session.job_id)
                    subject_lock = self._subject_session_locks.get(subject_key)
                    if subject_lock is not None and not subject_lock.locked():
                        self._subject_session_locks.pop(subject_key, None)

    def _register_session(self, session: SessionState) -> SessionState:
        self._sessions[session.session_id] = session
        if session.user_id and session.job_id:
            self._subject_session_ids[(session.user_id, session.job_id)] = session.session_id
        self._clean_expired_sessions()
        return session

    def remove_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session and session.user_id and session.job_id:
            subject_key = (session.user_id, session.job_id)
            if self._subject_session_ids.get(subject_key) == session_id:
                self._subject_session_ids.pop(subject_key, None)
        lock = self._recovery_locks.get(session_id)
        if lock is not None and not lock.locked():
            self._recovery_locks.pop(session_id, None)

    @staticmethod
    def build_subject_progress_snapshot(session: SessionState) -> dict[str, Any]:
        return build_subject_progress_snapshot(session)

    async def persist_subject_progress(self, session: SessionState) -> bool:
        if not session.job_id or not session.user_id:
            return True
        return await save_subject_progress_snapshot(
            session.job_id,
            session.user_id,
            self.build_subject_progress_snapshot(session),
        )

    async def create_session(
        self, max_steps: int = 9999, user_id: str | None = None
    ) -> SessionState:
        """Create a new learning session."""
        session_id = str(uuid.uuid4())[:8]

        # Default (Junyi) session: PCA over SAINT's trained concept embeddings.
        raw_emb = self._saint_model.concept_emb_matrix[1:].cpu().numpy()
        concept_embed_pca = apply_concept_pca(raw_emb)

        env = AdaptiveLearningEnv(
            saint_model=self._saint_model,
            concept_map=self._concept_map,
            prereq_graph=self._prereq_graph,
            max_steps=max_steps,
            mastery_threshold=self._mastery_threshold,
            deterministic_train=False,
            device=str(self._device),
            concept_embed_pca=concept_embed_pca,
        )
        obs, _info = env.reset(seed=42)

        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            env=env,
            q_net=self._q_net,
            device=self._device,
            concept_map=self._concept_map,
            concept_names=self._concept_names,
            concept_definitions=self._concept_defs,
            prereq_graph=self._prereq_graph,
            current_obs=obs,
            concept_theories={},
        )

        self._register_session(session)
        if not await self.persist_subject_progress(session):
            self.remove_session(session.session_id)
            raise RuntimeError(f"Failed to persist subject progress for {session.session_id}")
        return session

    def get_session(self, session_id: str) -> SessionState | None:
        session = self._sessions.get(session_id)
        if session:
            session.accessed_at = time.time()
        return session

    def get_active_pipeline_session(self, user_id: str, job_id: str) -> SessionState | None:
        session_id = self._subject_session_ids.get((user_id, job_id))
        session = self.get_session(session_id) if session_id else None
        return session if session and session.status == "active" else None

    def _build_external_embeddings(
        self,
        concept_map: dict[str, int],
        id_to_concept: dict[int, str],
        names: dict[str, str],
        defs: dict[str, str],
        precomputed_embeddings: list[list[float]] | None,
    ) -> Any:
        n = len(concept_map)
        if precomputed_embeddings is not None:
            logger.info(
                "[Session] Using precomputed embeddings ({} concepts)",
                len(precomputed_embeddings),
            )
            raw_emb = np.array(precomputed_embeddings, dtype=np.float32)
        else:
            ordered_texts = []
            for idx in range(n):
                cid = id_to_concept.get(idx, str(idx))
                name = names.get(cid, cid)
                definition = defs.get(cid, "")
                ordered_texts.append(f"{name}: {definition}" if definition else name)
            logger.info("[Session] Encoding {} new concepts...", n)
            raw_emb = self._encode_concepts(ordered_texts)
            logger.info("[Session] ✓ Embeddings shape: {}", raw_emb.shape)

        emb_dim = raw_emb.shape[1] if len(raw_emb.shape) > 1 else 0
        if emb_dim == 0:
            return None
        padded = np.zeros((n + 1, emb_dim), dtype=np.float32)
        padded[1 : n + 1] = raw_emb
        return torch.from_numpy(padded).to(self._device)

    async def _load_session_history(
        self,
        job_id: str | None,
        user_id: str | None,
        history_source_doc: dict[str, Any] | None,
    ) -> tuple[list[dict], int, int]:
        if history_source_doc:
            prev_history = history_source_doc.get("exercise_history") or []
            total_correct = history_source_doc.get("total_correct", 0)
            total_answered = history_source_doc.get("total_answered", 0)
            logger.info(
                "[Session] Rehydrating session from provided subject progress with {} exercises",
                len(prev_history),
            )
            return prev_history, total_correct, total_answered

        if not job_id:
            return [], 0, 0
        if not user_id:
            return [], 0, 0

        try:
            subject_progress = await load_subject_progress_for_user(job_id, user_id=user_id)
        except Exception:
            logger.exception("[Session] Error loading saved subject progress, starting fresh")
            return [], 0, 0

        if subject_progress and subject_progress.get("exercise_history"):
            prev_history = subject_progress["exercise_history"]
            logger.info(
                "[Session] Found saved subject progress with {} exercises for job {}",
                len(prev_history),
                job_id,
            )
            return (
                prev_history,
                subject_progress.get("total_correct", 0),
                subject_progress.get("total_answered", 0),
            )

        logger.info("[Session] No saved subject progress for job {}, starting fresh", job_id)
        return [], 0, 0

    async def create_session_from_pipeline(
        self,
        concepts_data: dict[str, Any],
        concept_map: dict[str, int],
        prereq_edges: list[dict],
        max_steps: int = 9999,
        precomputed_embeddings: list[list[float]] | None = None,
        job_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        history_source_doc: dict[str, Any] | None = None,
    ) -> "SessionState":
        """Create a learning session from PDF pipeline output."""
        session_id = session_id or str(uuid.uuid4())[:8]

        prereq_graph = self._build_prereq_graph_from_edges(prereq_edges, concept_map)
        names, defs, theories = self._build_concept_info_from_data(concepts_data, concept_map)
        id_to_concept = self._build_id_to_concept_map(concept_map)

        external_embeddings = self._build_external_embeddings(
            concept_map, id_to_concept, names, defs, precomputed_embeddings
        )

        # PCA features for the per-concept observation come from the same
        # 768-dim embeddings SAINT uses for this session (drop the PAD row).
        concept_embed_pca = None
        if external_embeddings is not None:
            raw_emb = external_embeddings[1:].cpu().numpy()
            concept_embed_pca = apply_concept_pca(raw_emb)

        env = AdaptiveLearningEnv(
            saint_model=self._saint_model,
            concept_map=concept_map,
            prereq_graph=prereq_graph,
            max_steps=max_steps,
            mastery_threshold=self._mastery_threshold,
            deterministic_train=False,
            device=str(self._device),
            external_embeddings=external_embeddings,
            concept_embed_pca=concept_embed_pca,
        )
        obs, _info = env.reset(seed=42)

        prev_history, total_correct, total_answered = await self._load_session_history(
            job_id, user_id, history_source_doc
        )

        if prev_history:
            env.inject_history(
                [ex["concept_idx"] for ex in prev_history],
                [ex["bloom_level"] for ex in prev_history],
                [1 if ex.get("is_correct") else 0 for ex in prev_history],
            )
            obs = env._build_obs()

        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            env=env,
            q_net=self._q_net,
            device=self._device,
            concept_map=concept_map,
            concept_names=names,
            concept_definitions=defs,
            prereq_graph=prereq_graph,
            current_obs=obs,
            total_correct=total_correct,
            total_answered=total_answered,
            job_id=job_id,
            created_at=(history_source_doc or {}).get("created_at", time.time()),
            status=(history_source_doc or {}).get("status", "active"),
            concept_theories=theories,
        )
        SessionState._restore_exercise_records(session, prev_history)

        self._register_session(session)
        if not await self.persist_subject_progress(session):
            self.remove_session(session.session_id)
            raise RuntimeError(f"Failed to persist subject progress for {session.session_id}")
        return session

    async def get_or_recover_session(
        self,
        session_id: str,
        user_id: str,
        *,
        session_doc: dict[str, Any] | None = None,
        job_doc: dict[str, Any] | None = None,
    ) -> SessionState | None:
        """Get active in-memory session; recover from MongoDB when missing.

        Recovery path:
        1) Load persisted session doc by session_id + user_id.
        2) Load source pipeline result by job_id.
        3) Rebuild environment and replay saved exercise history.
        """
        # Fast path
        active = self.get_session(session_id)
        if active and getattr(active, "user_id", None) == user_id:
            return active

        if session_id not in self._recovery_locks:
            self._recovery_locks[session_id] = asyncio.Lock()

        async with self._recovery_locks[session_id]:
            # Double check after acquiring lock
            active = self.get_session(session_id)
            if active and getattr(active, "user_id", None) == user_id:
                return active

            if session_doc is None:
                session_doc = await load_subject_progress_by_session_for_user(session_id, user_id)
            if not session_doc:
                return None

            if job_doc is None:
                job_id = session_doc.get("job_id")
                if not job_id:
                    logger.warning(
                        "[Session] Cannot recover session={}: missing job_id in persisted doc",
                        session_id,
                    )
                    return None
                job_doc = await load_pipeline_job_for_user(job_id, user_id)
            if not job_doc:
                logger.warning(
                    "[Session] Cannot recover session={}: pipeline job {} not found",
                    session_id,
                    session_doc.get("job_id"),
                )
                return None

            return await self._recover_session_from_documents(
                session_id=session_id,
                user_id=user_id,
                session_doc=session_doc,
                job_doc=job_doc,
            )

    async def _recover_session_from_documents(
        self,
        *,
        session_id: str,
        user_id: str,
        session_doc: dict[str, Any],
        job_doc: dict[str, Any],
    ) -> SessionState | None:
        job_id = session_doc.get("job_id")
        if not job_id:
            logger.warning(
                "[Session] Cannot recover session={}: missing job_id in persisted doc",
                session_id,
            )
            return None

        result = job_doc.get("result") or {}
        if not all(key in result for key in ("concepts_data", "concept_map", "prereq_edges")):
            logger.warning(
                "[Session] Cannot recover session={}: incomplete pipeline result", session_id
            )
            return None

        max_steps = int(session_doc.get("max_steps") or _MIN_STORED_MAX_STEPS)
        if max_steps <= _MIN_STORED_MAX_STEPS:
            max_steps = _DEFAULT_MAX_STEPS

        try:
            recovered = await self.create_session_from_pipeline(
                concepts_data=result["concepts_data"],
                concept_map=result["concept_map"],
                prereq_edges=result["prereq_edges"],
                max_steps=max_steps,
                precomputed_embeddings=result.get("concept_embeddings"),
                job_id=job_id,
                user_id=user_id,
                session_id=session_id,
                history_source_doc=session_doc,
            )
            logger.info(
                "[Session] Recovered session={} for user={} from Mongo", session_id, user_id
            )
        except Exception:
            logger.exception("[Session] Failed recovering session={}", session_id)
            return None
        else:
            return recovered

    async def get_or_create_pipeline_session(
        self,
        *,
        job_doc: dict[str, Any],
        subject_progress: dict[str, Any] | None,
        user_id: str,
        max_steps: int,
    ) -> tuple[SessionState, bool]:
        """Return one active subject session, recovering or creating it atomically."""
        job_id = str(job_doc["job_id"])
        result = job_doc["result"]
        subject_key = (user_id, job_id)
        lock = self._subject_session_locks.setdefault(subject_key, asyncio.Lock())

        async with lock:
            active_session_id = self._subject_session_ids.get(subject_key)
            if active_session_id:
                active_session = self.get_session(active_session_id)
                if active_session and active_session.status == "active":
                    return active_session, False

            last_session_id = subject_progress.get("last_session_id") if subject_progress else None
            if (
                subject_progress
                and subject_progress.get("status") == "active"
                and isinstance(last_session_id, str)
                and last_session_id.strip()
            ):
                existing = await self.get_or_recover_session(
                    last_session_id.strip(),
                    user_id,
                    session_doc=subject_progress,
                    job_doc=job_doc,
                )
                if existing and existing.job_id == job_id and existing.status == "active":
                    return existing, False

            history_source_doc = None
            if subject_progress:
                history_source_doc = {**subject_progress, "status": "active"}

            session = await self.create_session_from_pipeline(
                concepts_data=result["concepts_data"],
                concept_map=result["concept_map"],
                prereq_edges=result["prereq_edges"],
                max_steps=max_steps,
                precomputed_embeddings=result.get("concept_embeddings"),
                job_id=job_id,
                user_id=user_id,
                history_source_doc=history_source_doc,
            )
            return session, True

    # ── Query Methods ───────────────────────────────────────

    def get_session_status(self, session_id: str) -> dict[str, Any] | None:
        """Get full session status."""
        session = self.get_session(session_id)
        if not session:
            return None

        env_stats = session.env.get_session_stats()
        concept_mastery = session.env.get_concept_mastery()
        unlocked_mask = session.env.get_prereq_ok_mask(threshold=self._mastery_threshold)
        progress_metrics = summarize_mastery_progress(
            concept_mastery=concept_mastery,
            unlocked_mask=unlocked_mask,
            threshold=self._mastery_threshold,
        )

        return {
            "session_id": session_id,
            "status": session.status,
            "step": env_stats["step"],
            "max_steps": env_stats["max_steps"],
            "concepts_visited": env_stats["concepts_visited"],
            "total_concepts": progress_metrics["total_concepts"],
            "unlocked_concepts": progress_metrics["unlocked_concepts"],
            "locked_concepts": progress_metrics["locked_concepts"],
            "mastered_concepts": progress_metrics["mastered_concepts"],
            "avg_mastery": progress_metrics["avg_mastery"],
            "progress_percent": progress_metrics["progress_percent"],
            "coverage": env_stats["coverage"],
            "total_correct": session.total_correct,
            "total_answered": session.total_answered,
            "accuracy": session.total_correct / max(session.total_answered, 1),
            "exercises": [
                {
                    "exercise_id": ex.exercise_id,
                    "concept_name": ex.concept_name,
                    "bloom_level": ex.bloom_level,
                    "is_correct": ex.is_correct,
                }
                for ex in session.exercise_history
            ],
        }

    @staticmethod
    def _resolve_concept_status(mastery: float, *, visited: bool, prereq_ok: bool) -> str:
        """Status priority: prerequisite lock > mastered > in-progress > available."""
        if not prereq_ok:
            return "locked"
        if mastery >= _MASTERY_THRESHOLD:
            return "mastered"
        if visited:
            return "in_progress"
        return "available"

    def get_knowledge_graph(self, session_id: str) -> dict[str, Any] | None:
        """Get knowledge graph with mastery overlay."""
        session = self.get_session(session_id)
        if not session:
            return None

        concept_mastery = session.env.get_concept_mastery()
        id_to_concept = {v: k for k, v in session.concept_map.items()}
        prereq_ok_mask = session.env.get_prereq_ok_mask(threshold=self._mastery_threshold)

        nodes = []
        for idx in range(len(session.concept_map)):
            cid = id_to_concept.get(idx, str(idx))
            mastery = float(concept_mastery[idx])
            visited = session.env.is_concept_visited(idx)
            status = self._resolve_concept_status(
                mastery,
                visited=visited,
                prereq_ok=bool(prereq_ok_mask[idx]),
            )
            nodes.append(
                {
                    "id": cid,
                    "index": idx,
                    "name": session.concept_names.get(cid, cid),
                    "mastery": mastery,
                    "status": status,
                    "visited": visited,
                }
            )

        edges = []
        for tgt_idx, src_list in session.prereq_graph.items():
            tgt_id = id_to_concept.get(tgt_idx, str(tgt_idx))
            for src_idx in src_list:
                src_id = id_to_concept.get(src_idx, str(src_idx))
                edges.append({"source": src_id, "target": tgt_id})

        return {"nodes": nodes, "edges": edges}

    def get_mastery_matrix(self, session_id: str) -> dict[str, Any] | None:
        """Get full mastery matrix (concepts x bloom levels)."""
        session = self.get_session(session_id)
        if not session:
            return None

        bloom_mastery = session.env.get_mastery_matrix()
        id_to_concept = {v: k for k, v in session.concept_map.items()}

        matrix = []
        for idx in range(len(session.concept_map)):
            cid = id_to_concept.get(idx, str(idx))
            matrix.append(
                {
                    "concept_id": cid,
                    "concept_name": session.concept_names.get(cid, cid),
                    "bloom_levels": [float(bloom_mastery[idx, b]) for b in range(6)],
                }
            )

        return {
            "matrix": matrix,
            "bloom_labels": list(BLOOM_LABEL_SEQUENCE),
        }

    def get_concept_detail(self, session_id: str, concept_id: str) -> dict[str, Any] | None:
        """Get detailed info for a specific concept."""
        session = self.get_session(session_id)
        if not session or concept_id not in session.concept_map:
            return None

        idx = session.concept_map[concept_id]
        concept_mastery = session.env.get_concept_mastery()
        bloom_mastery = session.env.get_mastery_matrix()
        prereq_ok_mask = session.env.get_prereq_ok_mask(threshold=self._mastery_threshold)
        id_to_concept = {v: k for k, v in session.concept_map.items()}

        prereqs = [
            {
                "id": id_to_concept.get(p, str(p)),
                "name": session.concept_names.get(id_to_concept.get(p, str(p)), str(p)),
                "mastery": float(concept_mastery[p]),
            }
            for p in session.prereq_graph.get(idx, [])
        ]

        dependents = []
        for tgt_idx, src_list in session.prereq_graph.items():
            if idx in src_list:
                tgt_id = id_to_concept.get(tgt_idx, str(tgt_idx))
                dependents.append(
                    {
                        "id": tgt_id,
                        "name": session.concept_names.get(tgt_id, tgt_id),
                        "mastery": float(concept_mastery[tgt_idx]),
                    }
                )

        return {
            "id": concept_id,
            "name": session.concept_names.get(concept_id, concept_id),
            "definition": session.concept_definitions.get(concept_id, ""),
            "mastery": float(concept_mastery[idx]),
            "status": self._resolve_concept_status(
                mastery=float(concept_mastery[idx]),
                visited=session.env.is_concept_visited(idx),
                prereq_ok=bool(prereq_ok_mask[idx]),
            ),
            "bloom_mastery": [float(bloom_mastery[idx, b]) for b in range(6)],
            "prerequisites": prereqs,
            "dependents": dependents,
            "visited": session.env.is_concept_visited(idx),
            "visit_count": session.env.get_visit_count(idx),
        }
