"""
session.py — Session state management (SessionManager + SessionState).

SessionManager handles:
  - Model loading (SAINT + DQN)
  - Session lifecycle (create, get, status)
  - Knowledge graph and mastery queries

Exercise-related logic (generation, submission, prefetch) is delegated
to services/exercise_service.py.
"""

import uuid
import time
import asyncio
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field

import numpy as np
import torch
from loguru import logger

from .models import load_saint_model, load_dqn_model, DuelingQNetwork, SaintModel
from .environment import AdaptiveLearningEnv
from . import mongo_store


@dataclass
class ExerciseRecord:
    exercise_id: str
    concept_idx: int
    concept_name: str
    bloom_level: int
    question: str
    options: Dict[str, str]
    correct_option: str
    explanation: str
    explanation_correct: str = ""
    explanation_incorrect: str = ""
    theory: Optional[Dict[str, Any]] = None
    user_answer: Optional[str] = None
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
    status: str = "active"
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    concept_theories: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _prefetch_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)


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

        self._concepts_data = concepts_data or {}
        self._prereq_data = prereq_data or []
        self._prereq_graph = self._build_prereq_graph()
        self._concept_names, self._concept_defs = self._build_concept_info()

        # Active sessions
        self._sessions: Dict[str, SessionState] = {}

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
        graph: Dict[int, List[int]] = {}
        dropped = 0
        for edge in self._prereq_data:
            src = str(edge.get("source", "")).strip()
            tgt = str(edge.get("target", "")).strip()
            if src in self._concept_map and tgt in self._concept_map:
                tgt_idx = self._concept_map[tgt]
                src_idx = self._concept_map[src]
                graph.setdefault(tgt_idx, []).append(src_idx)
            else:
                dropped += 1
        if dropped:
            logger.warning(f"[Session] _build_prereq_graph: dropped {dropped} edges")
        return graph

    def _build_concept_info(self):
        names: Dict[str, str] = {}
        defs: Dict[str, str] = {}
        id_to_concept = {v: k for k, v in self._concept_map.items()}

        for cid, cdata in self._concepts_data.items():
            name = cdata.get("name", cid)
            if not name or str(name).lower() == "nan":
                name = cid
            names[cid] = name
            defs[cid] = cdata.get("definition", "")

        for idx in range(len(self._concept_map)):
            cid = id_to_concept.get(idx, str(idx))
            if cid not in names:
                names[cid] = cid
                defs[cid] = ""

        return names, defs

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

    def create_session(self, max_steps: int = 50, user_id: Optional[str] = None) -> SessionState:
        """Create a new learning session."""
        session_id = str(uuid.uuid4())[:8]

        env = AdaptiveLearningEnv(
            saint_model=self._saint_model,
            concept_map=self._concept_map,
            prereq_graph=self._prereq_graph,
            max_steps=max_steps,
            mastery_threshold=0.75,
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

        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    async def create_session_from_pipeline(
        self,
        concepts_data: Dict[str, Any],
        concept_map: Dict[str, int],
        prereq_edges: List[Dict],
        max_steps: int = 50,
        precomputed_embeddings: Optional[List[List[float]]] = None,
        job_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> SessionState:
        """Create a learning session from PDF pipeline output."""
        session_id = str(uuid.uuid4())[:8]

        # Build prereq graph
        prereq_graph: Dict[int, List[int]] = {}
        dropped_edges = 0
        for edge in prereq_edges:
            src = str(edge.get("source", "")).strip()
            tgt = str(edge.get("target", "")).strip()
            if src in concept_map and tgt in concept_map:
                tgt_idx = concept_map[tgt]
                src_idx = concept_map[src]
                prereq_graph.setdefault(tgt_idx, []).append(src_idx)
            else:
                dropped_edges += 1
        if dropped_edges:
            logger.warning(f"[Session] create_session_from_pipeline: dropped {dropped_edges}/{len(prereq_edges)} edges")

        # Build names/defs
        names: Dict[str, str] = {}
        defs: Dict[str, str] = {}
        theories: Dict[str, Dict[str, Any]] = {}
        id_to_concept = {v: k for k, v in concept_map.items()}
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
            mastery_threshold=0.75,
            deterministic_train=False,
            device=str(self._device),
            external_embeddings=external_embeddings,
        )
        obs, info = env.reset(seed=42)

        # Load previous history from MongoDB
        prev_history = []
        total_correct = 0
        total_answered = 0
        if job_id:
            try:
                prev_session = await mongo_store.find_latest_session_for_job(job_id, user_id=user_id)
                if prev_session and prev_session.get("exercise_history"):
                    prev_history = prev_session["exercise_history"]
                    total_correct = prev_session.get("total_correct", 0)
                    total_answered = prev_session.get("total_answered", 0)
                    logger.info(f"[Session] Found previous session with {len(prev_history)} exercises")
                else:
                    logger.info(f"[Session] No previous session for job {job_id}, starting fresh")
            except Exception as e:
                logger.warning(f"[Session] Error loading previous session: {e}, starting fresh")

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
                options=ex.get("options", {}),
                correct_option=ex.get("correct_option", ""),
                explanation=ex.get("explanation", ""),
                explanation_correct=ex.get("explanation_correct", ""),
                explanation_incorrect=ex.get("explanation_incorrect", ""),
                user_answer=ex.get("user_answer"),
                is_correct=ex.get("is_correct"),
                timestamp=ex.get("timestamp", 0),
            ))

        self._sessions[session_id] = session
        return session

    # ── Query Methods ───────────────────────────────────────

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get full session status."""
        session = self._sessions.get(session_id)
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
        if mastery >= 0.75:
            return "mastered"
        if visited:
            return "in_progress"
        return "available"

    def get_knowledge_graph(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge graph with mastery overlay."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        concept_mastery = session.env.get_concept_mastery()
        id_to_concept = {v: k for k, v in session.concept_map.items()}
        prereq_ok_mask = session.env.get_prereq_ok_mask(threshold=0.75)

        nodes = []
        for idx in range(len(session.concept_map)):
            cid = id_to_concept.get(idx, str(idx))
            mastery = float(concept_mastery[idx])
            status = self._resolve_concept_status(
                mastery=mastery,
                visited=(idx in session.env._visited),
                prereq_ok=bool(prereq_ok_mask[idx]),
            )
            nodes.append({
                "id": cid,
                "index": idx,
                "name": session.concept_names.get(cid, cid),
                "mastery": mastery,
                "status": status,
                "visited": idx in session.env._visited,
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
        session = self._sessions.get(session_id)
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
        session = self._sessions.get(session_id)
        if not session or concept_id not in session.concept_map:
            return None

        idx = session.concept_map[concept_id]
        concept_mastery = session.env.get_concept_mastery()
        bloom_mastery = session.env.get_mastery_matrix()
        prereq_ok_mask = session.env.get_prereq_ok_mask(threshold=0.75)
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
                visited=(idx in session.env._visited),
                prereq_ok=bool(prereq_ok_mask[idx]),
            ),
            "bloom_mastery": [float(bloom_mastery[idx, b]) for b in range(6)],
            "prerequisites": prereqs,
            "dependents": dependents,
            "visited": idx in session.env._visited,
            "visit_count": session.env._visit_counts.get(idx, 0),
        }
