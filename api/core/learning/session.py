"""
session.py — Session state management (SessionManager + SessionState).

SessionManager handles:
  - Model loading (SAINT + DQN)
  - Session lifecycle (create, get, status)
  - Knowledge graph and mastery queries

Exercise-related logic (generation, submission, prefetch) is delegated
to the learning exercise service.
"""

import uuid
import time
import asyncio
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

import numpy as np
import torch
from loguru import logger

from ...config import get_settings
from .models import load_saint_model, load_dqn_model, DuelingQNetwork, SaintModel
from .environment import AdaptiveLearningEnv
from ..shared import mongo_store
from .subject_progress_snapshot import build_subject_progress_snapshot

_MASTERY_THRESHOLD = float(get_settings().adaptive_mastery_threshold)


@dataclass
class ExerciseRecord:
    exercise_id: str
    concept_idx: int
    concept_name: str
    bloom_level: int
    question: str
    correct_option: str
    explanation: str
    exercise_type: str = "mcq"
    options: Dict[str, str] = field(default_factory=dict)
    statement: Optional[str] = None
    hint: Optional[str] = None
    items: List[str] = field(default_factory=list)
    pairs: List[Dict[str, str]] = field(default_factory=list)
    right_items: List[str] = field(default_factory=list)
    rubric: List[str] = field(default_factory=list)
    correct_answer: Any = None
    explanation_correct: str = ""
    explanation_incorrect: str = ""
    theory: Optional[Dict[str, Any]] = None
    user_answer: Optional[Any] = None
    is_correct: Optional[bool] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionState:
    session_id: str
    env: AdaptiveLearningEnv
    q_net: DuelingQNetwork
    device: torch.device
    concept_map: Dict[str, int]
    concept_names: Dict[str, str]
    concept_definitions: Dict[str, str]
    prereq_graph: Dict[int, List[int]]
    current_obs: np.ndarray
    user_id: Optional[str] = None
    current_exercise: Optional[ExerciseRecord] = None
    exercise_history: List[ExerciseRecord] = field(default_factory=list)
    total_correct: int = 0
    total_answered: int = 0
    job_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    status: str = "active"
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    concept_theories: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _prefetch_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tutor_chat_history: List[Dict[str, str]] = field(default_factory=list)
    tutor_chat_exercise_id: Optional[str] = None


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
        concepts_data: Optional[Dict] = None,
        prereq_data: Optional[List[Dict]] = None,
        device: Optional[str] = None,
    ):
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._saint_path = saint_path
        self._dqn_path = dqn_path

        # Load models once
        self._saint_model, self._concept_map, self._saint_config = load_saint_model(saint_path, self._device)
        self._q_net, self._dqn_info = load_dqn_model(dqn_path, self._device)
        self._mastery_threshold = _MASTERY_THRESHOLD

        self._concepts_data = concepts_data or {}
        self._prereq_data = prereq_data or []
        self._prereq_graph = self._build_prereq_graph()
        self._concept_names, self._concept_defs = self._build_concept_info()

        # Active sessions
        self._sessions: Dict[str, SessionState] = {}
        self._recovery_locks: Dict[str, asyncio.Lock] = {}

    # ── Properties ──────────────────────────────────────────

    @property
    def concept_map(self):
        return self._concept_map

    @property
    def concept_names(self):
        return self._concept_names

    @property
    def n_concepts(self):
        return len(self._concept_map)

    # ── Internal helpers ────────────────────────────────────

    def _build_prereq_graph(self) -> Dict[int, List[int]]:
        return self._build_prereq_graph_from_edges(self._prereq_data, self._concept_map)

    @staticmethod
    def _build_id_to_concept_map(concept_map: Dict[str, int]) -> Dict[int, str]:
        return {v: k for k, v in concept_map.items()}

    @classmethod
    def _build_prereq_graph_from_edges(
        cls,
        prereq_edges: List[Dict],
        concept_map: Dict[str, int],
    ) -> Dict[int, List[int]]:
        graph: Dict[int, List[int]] = {}
        dropped = 0
        for edge in prereq_edges:
            src = str(edge.get("source", "")).strip()
            tgt = str(edge.get("target", "")).strip()
            if src in concept_map and tgt in concept_map:
                tgt_idx = concept_map[tgt]
                src_idx = concept_map[src]
                graph.setdefault(tgt_idx, []).append(src_idx)
            else:
                dropped += 1
        if dropped:
            logger.warning(
                f"[Session] _build_prereq_graph_from_edges: dropped {dropped}/{len(prereq_edges)} edges"
            )
        return graph

    def _build_concept_info(self):
        names, defs, _ = self._build_concept_info_from_data(self._concepts_data, self._concept_map)
        return names, defs

    @classmethod
    def _build_concept_info_from_data(
        cls,
        concepts_data: Dict[str, Any],
        concept_map: Dict[str, int],
    ):
        names: Dict[str, str] = {}
        defs: Dict[str, str] = {}
        theories: Dict[str, Dict[str, Any]] = {}
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
    def _get_text_encoder(cls):
        """Lazy-load sentence-transformer (same model used in SAINT training)."""
        if cls._text_encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[Session] Loading text encoder: {cls._TEXT_MODEL_NAME}")
            cls._text_encoder = SentenceTransformer(cls._TEXT_MODEL_NAME)
            logger.info("[Session] ✓ Text encoder ready")
        return cls._text_encoder

    def _encode_concepts(self, concept_names_ordered: List[str]) -> np.ndarray:
        """Encode concept names into 768d embeddings."""
        encoder = self._get_text_encoder()
        embeddings = encoder.encode(concept_names_ordered, show_progress_bar=False, batch_size=32)
        return np.array(embeddings, dtype=np.float32)

    # ── Session Lifecycle ───────────────────────────────────

    def _clean_expired_sessions(self, max_size=500):
        """Simple eviction logic: if we exceed max_size, evict oldest 20% by access time."""
        if len(self._sessions) > max_size:
            sorted_keys = sorted(
                self._sessions.keys(),
                key=lambda k: getattr(self._sessions[k], 'accessed_at', getattr(self._sessions[k], 'created_at', 0))
            )
            # Remove oldest 20%
            for k in sorted_keys[:max_size // 5]:
                self._sessions.pop(k, None)
                lock = self._recovery_locks.get(k)
                if lock is not None and not lock.locked():
                    self._recovery_locks.pop(k, None)

    def _register_session(self, session: SessionState) -> SessionState:
        self._sessions[session.session_id] = session
        self._clean_expired_sessions()
        return session

    def remove_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        lock = self._recovery_locks.get(session_id)
        if lock is not None and not lock.locked():
            self._recovery_locks.pop(session_id, None)

    @staticmethod
    def build_subject_progress_snapshot(session: SessionState) -> Dict[str, Any]:
        return build_subject_progress_snapshot(session)

    async def persist_subject_progress(self, session: SessionState) -> bool:
        if not session.job_id or not session.user_id:
            return True
        return await mongo_store.save_subject_progress(
            session.job_id,
            session.user_id,
            self.build_subject_progress_snapshot(session),
        )

    async def create_session(self, max_steps: int = 9999, user_id: Optional[str] = None) -> SessionState:
        """Create a new learning session."""
        session_id = str(uuid.uuid4())[:8]

        env = AdaptiveLearningEnv(
            saint_model=self._saint_model,
            concept_map=self._concept_map,
            prereq_graph=self._prereq_graph,
            max_steps=max_steps,
            mastery_threshold=self._mastery_threshold,
            deterministic_train=False,
            device=str(self._device),
        )
        obs, info = env.reset(seed=42)

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

    def get_session(self, session_id: str) -> Optional[SessionState]:
        session = self._sessions.get(session_id)
        if session:
            session.accessed_at = time.time()
        return session

    async def create_session_from_pipeline(
        self,
        concepts_data: Dict[str, Any],
        concept_map: Dict[str, int],
        prereq_edges: List[Dict],
        max_steps: int = 9999,
        precomputed_embeddings: Optional[List[List[float]]] = None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        history_source_doc: Optional[Dict[str, Any]] = None,
    ) -> SessionState:
        """Create a learning session from PDF pipeline output."""
        session_id = session_id or str(uuid.uuid4())[:8]

        prereq_graph = self._build_prereq_graph_from_edges(prereq_edges, concept_map)

        names, defs, theories = self._build_concept_info_from_data(concepts_data, concept_map)
        id_to_concept = self._build_id_to_concept_map(concept_map)

        # Build concept embeddings for SAINT
        n_pipeline = len(concept_map)
        if precomputed_embeddings is not None:
            logger.info(f"[Session] Using precomputed embeddings ({len(precomputed_embeddings)} concepts)")
            raw_emb = np.array(precomputed_embeddings, dtype=np.float32)
        else:
            ordered_names = []
            for idx in range(n_pipeline):
                cid = id_to_concept.get(idx, str(idx))
                name = names.get(cid, cid)
                definition = defs.get(cid, "")
                text = f"{name}: {definition}" if definition else name
                ordered_names.append(text)
            logger.info(f"[Session] Encoding {n_pipeline} new concepts...")
            raw_emb = self._encode_concepts(ordered_names)
            logger.info(f"[Session] ✓ Embeddings shape: {raw_emb.shape}")

        emb_dim = raw_emb.shape[1] if len(raw_emb.shape) > 1 else 0
        if emb_dim > 0:
            padded = np.zeros((n_pipeline + 1, emb_dim), dtype=np.float32)
            padded[1:n_pipeline + 1] = raw_emb
            external_embeddings = torch.from_numpy(padded).to(self._device)
        else:
            external_embeddings = None

        env = AdaptiveLearningEnv(
            saint_model=self._saint_model,
            concept_map=concept_map,
            prereq_graph=prereq_graph,
            max_steps=max_steps,
            mastery_threshold=self._mastery_threshold,
            deterministic_train=False,
            device=str(self._device),
            external_embeddings=external_embeddings,
        )
        obs, info = env.reset(seed=42)

        # Load saved subject progress from MongoDB
        prev_history = []
        total_correct = 0
        total_answered = 0
        if history_source_doc:
            prev_history = history_source_doc.get("exercise_history") or []
            total_correct = history_source_doc.get("total_correct", 0)
            total_answered = history_source_doc.get("total_answered", 0)
            logger.info(
                f"[Session] Rehydrating session from provided subject progress with {len(prev_history)} exercises"
            )
        elif job_id:
            try:
                subject_progress = await mongo_store.load_subject_progress_for_job(job_id, user_id=user_id)
                if subject_progress and subject_progress.get("exercise_history"):
                    prev_history = subject_progress["exercise_history"]
                    total_correct = subject_progress.get("total_correct", 0)
                    total_answered = subject_progress.get("total_answered", 0)
                    logger.info(
                        f"[Session] Found saved subject progress with {len(prev_history)} exercises for job {job_id}"
                    )
                else:
                    logger.info(f"[Session] No saved subject progress for job {job_id}, starting fresh")
            except Exception as e:
                logger.warning(f"[Session] Error loading saved subject progress: {e}, starting fresh")

        # Inject history into environment
        if prev_history:
            concept_indices = [ex["concept_idx"] for ex in prev_history]
            bloom_levels = [ex["bloom_level"] for ex in prev_history]
            responses = [1 if ex.get("is_correct") else 0 for ex in prev_history]
            env.inject_history(concept_indices, bloom_levels, responses)
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

        # Restore exercise history records
        for ex in prev_history:
            session.exercise_history.append(ExerciseRecord(
                exercise_id=ex.get("exercise_id", ""),
                concept_idx=ex["concept_idx"],
                concept_name=ex.get("concept_name", ""),
                bloom_level=ex["bloom_level"],
                question=ex.get("question", ""),
                correct_option=ex.get("correct_option", ""),
                explanation=ex.get("explanation", ""),
                exercise_type=ex.get("exercise_type", "mcq"),
                options=ex.get("options", {}),
                statement=ex.get("statement"),
                hint=ex.get("hint"),
                items=ex.get("items", []),
                pairs=ex.get("pairs", []),
                right_items=ex.get("right_items", []),
                rubric=ex.get("rubric", []),
                correct_answer=ex.get("correct_answer"),
                explanation_correct=ex.get("explanation_correct", ""),
                explanation_incorrect=ex.get("explanation_incorrect", ""),
                user_answer=ex.get("user_answer"),
                is_correct=ex.get("is_correct"),
                timestamp=ex.get("timestamp", 0),
            ))

        self._register_session(session)
        if not await self.persist_subject_progress(session):
            self.remove_session(session.session_id)
            raise RuntimeError(f"Failed to persist subject progress for {session.session_id}")
        return session

    async def get_or_recover_session(
        self,
        session_id: str,
        user_id: str,
    ) -> Optional[SessionState]:
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

            session_doc = await mongo_store.load_session_doc_for_user(session_id, user_id)
            if not session_doc:
                return None

        job_id = session_doc.get("job_id")
        if not job_id:
            logger.warning(
                f"[Session] Cannot recover session={session_id}: missing job_id in persisted doc"
            )
            return None

        job_doc = await mongo_store.load_pipeline_job_for_user(job_id, user_id)
        if not job_doc:
            logger.warning(
                f"[Session] Cannot recover session={session_id}: pipeline job {job_id} not found"
            )
            return None

        result = job_doc.get("result") or {}
        if not all(key in result for key in ("concepts_data", "concept_map", "prereq_edges")):
            logger.warning(
                f"[Session] Cannot recover session={session_id}: incomplete pipeline result"
            )
            return None

        max_steps = int(session_doc.get("max_steps") or 50)
        if max_steps <= 50:
            max_steps = 9999

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
                f"[Session] Recovered session={session_id} for user={user_id} from Mongo"
            )
            return recovered
        except Exception as exc:
            logger.warning(
                f"[Session] Failed recovering session={session_id}: {exc}"
            )
            return None

    # ── Query Methods ───────────────────────────────────────

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get full session status."""
        session = self.get_session(session_id)
        if not session:
            return None

        env_stats = session.env.get_session_stats()

        return {
            "session_id": session_id,
            "status": session.status,
            "step": env_stats["step"],
            "max_steps": env_stats["max_steps"],
            "concepts_visited": env_stats["concepts_visited"],
            "total_concepts": env_stats["total_concepts"],
            "avg_mastery": env_stats["avg_mastery"],
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
    def _resolve_concept_status(mastery: float, visited: bool, prereq_ok: bool) -> str:
        """Status priority: prerequisite lock > mastered > in-progress > available."""
        if not prereq_ok:
            return "locked"
        if mastery >= _MASTERY_THRESHOLD:
            return "mastered"
        if visited:
            return "in_progress"
        return "available"

    def get_knowledge_graph(self, session_id: str) -> Optional[Dict[str, Any]]:
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
                mastery=mastery,
                visited=visited,
                prereq_ok=bool(prereq_ok_mask[idx]),
            )
            nodes.append({
                "id": cid,
                "index": idx,
                "name": session.concept_names.get(cid, cid),
                "mastery": mastery,
                "status": status,
                "visited": visited,
            })

        edges = []
        for tgt_idx, src_list in session.prereq_graph.items():
            tgt_id = id_to_concept.get(tgt_idx, str(tgt_idx))
            for src_idx in src_list:
                src_id = id_to_concept.get(src_idx, str(src_idx))
                edges.append({"source": src_id, "target": tgt_id})

        return {"nodes": nodes, "edges": edges}

    def get_mastery_matrix(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get full mastery matrix (concepts × bloom levels)."""
        session = self.get_session(session_id)
        if not session:
            return None

        bloom_mastery = session.env.get_mastery_matrix()
        id_to_concept = {v: k for k, v in session.concept_map.items()}

        matrix = []
        for idx in range(len(session.concept_map)):
            cid = id_to_concept.get(idx, str(idx))
            matrix.append({
                "concept_id": cid,
                "concept_name": session.concept_names.get(cid, cid),
                "bloom_levels": [float(bloom_mastery[idx, b]) for b in range(6)],
            })

        return {"matrix": matrix, "bloom_labels": [
            "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"
        ]}

    def get_concept_detail(self, session_id: str, concept_id: str) -> Optional[Dict[str, Any]]:
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
                dependents.append({
                    "id": tgt_id,
                    "name": session.concept_names.get(tgt_id, tgt_id),
                    "mastery": float(concept_mastery[tgt_idx]),
                })

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
