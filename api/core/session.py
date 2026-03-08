"""
session.py — Session state management
"""

import uuid
import time
import asyncio
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .models import load_saint_model, load_dqn_model, DuelingQNetwork, SaintModel
from .environment import AdaptiveLearningEnv
from .agent import select_action, decode_action
from .exercise_gen import generate_exercise, evaluate_answer, generate_theory
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
    current_exercise: Optional[ExerciseRecord] = None
    exercise_history: List[ExerciseRecord] = field(default_factory=list)
    total_correct: int = 0
    total_answered: int = 0
    job_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    status: str = "active"  # active, completed
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    concept_theories: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class SessionManager:
    """In-memory session storage and lifecycle management."""

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

        # Store concept metadata
        self._concepts_data = concepts_data or {}
        self._prereq_data = prereq_data or []

        # Build prereq graph (concept_idx → list of prereq_idx)
        self._prereq_graph = self._build_prereq_graph()

        # Build concept names/definitions lookup
        self._concept_names, self._concept_defs = self._build_concept_info()

        # Active sessions
        self._sessions: Dict[str, SessionState] = {}

        # Thread pool for blocking LLM calls
        self._llm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")

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
            print(f"[Session] _build_prereq_graph: dropped {dropped} edges due to unresolved concept IDs")
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

        # Ensure all concepts have entries
        for idx in range(len(self._concept_map)):
            cid = id_to_concept.get(idx, str(idx))
            if cid not in names:
                names[cid] = cid
                defs[cid] = ""

        return names, defs

    def create_session(self, max_steps: int = 50) -> SessionState:
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

    async def get_next_concept(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Step 1: Use D3QN to select next concept+bloom. Returns concept info and bloom level."""
        session = self._sessions.get(session_id)
        if not session or session.status != "active":
            return None

        async with session._lock:
            env = session.env
            env_stats = env.get_session_stats()
            current_step = env_stats.get("step", 0)

            # ── Force first step: always concept 0 + Bloom 1 ───────────────────
            if current_step == 0:
                concept_idx = 0
                bloom_level = 1
                action_id = concept_idx * 6 + (bloom_level - 1)
                print(f"\n{'═'*60}")
                print(f"[Session] 🎯 STEP 0 — Forcing warm-up: concept_idx=0, bloom=1")
            else:
                masks = env.action_masks()
                action_id = select_action(
                    session.q_net, session.current_obs, masks, session.device,
                    n_concepts=env.n_concepts,
                )
                concept_idx, bloom_level = decode_action(action_id)
                print(f"\n{'═'*60}")
                print(f"[Session] 🤖 D3QN selected action_id={action_id}")

            id_to_concept = {v: k for k, v in session.concept_map.items()}
            concept_id = id_to_concept.get(concept_idx, str(concept_idx))
            concept_name = session.concept_names.get(concept_id, concept_id)
            concept_def = session.concept_definitions.get(concept_id, "")

            BLOOM_NAMES = {
                1: "Remember", 2: "Understand", 3: "Apply",
                4: "Analyze",  5: "Evaluate",  6: "Create",
            }
            print(f"[Session] Concept : [{concept_idx}] {concept_name}")
            print(f"[Session] Bloom   : Level {bloom_level} ({BLOOM_NAMES.get(bloom_level, '?')})")
            print(f"[Session] Step    : {current_step + 1}/{env_stats.get('max_steps', '?')}")
            print(f"{'─'*60}")

            session._pending_concept_idx = concept_idx
            session._pending_bloom_level = bloom_level
            session._pending_action = action_id
            session.current_exercise = None  # Clear previous

            return {
                "concept_name": concept_name,
                "concept_idx": concept_idx,
                "bloom_level": bloom_level,
                "bloom_label": BLOOM_NAMES.get(bloom_level, "Unknown"),
                "step": env_stats["step"],
                "max_steps": env_stats["max_steps"],
            }

    async def get_theory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Step 2 (Optional): Generate theory if Bloom <= 2 based on pending concept."""
        session = self._sessions.get(session_id)
        if not session or not hasattr(session, '_pending_concept_idx'):
            return None

        concept_idx = session._pending_concept_idx
        id_to_concept = {v: k for k, v in session.concept_map.items()}
        concept_id = id_to_concept.get(concept_idx, str(concept_idx))
        
        # Check if theory was pre-generated
        if concept_id in session.concept_theories and session.concept_theories[concept_id]:
            return session.concept_theories[concept_id]

        concept_name = session.concept_names.get(concept_id, concept_id)
        concept_def = session.concept_definitions.get(concept_id, "")

        print(f"[Session] Fetching theory for {concept_name}...")
        loop = asyncio.get_event_loop()
        theory_data = await loop.run_in_executor(
            self._llm_executor,
            generate_theory, concept_name, concept_def,
        )
        
        # Cache for future queries
        session.concept_theories[concept_id] = theory_data
        
        return theory_data

    async def generate_exercise(self, session_id: str) -> Optional[ExerciseRecord]:
        """Step 3: Generate the actual exercise based on pending concept and bloom."""
        session = self._sessions.get(session_id)
        if not session or not hasattr(session, '_pending_concept_idx'):
            return None

        concept_idx = session._pending_concept_idx
        bloom_level = session._pending_bloom_level

        id_to_concept = {v: k for k, v in session.concept_map.items()}
        concept_id = id_to_concept.get(concept_idx, str(concept_idx))
        concept_name = session.concept_names.get(concept_id, concept_id)
        concept_def = session.concept_definitions.get(concept_id, "")

        print(f"[Session] Generating exercise for {concept_name} (Bloom {bloom_level})...")
        loop = asyncio.get_event_loop()
        exercise_data = await loop.run_in_executor(
            self._llm_executor,
            generate_exercise, concept_name, concept_def, bloom_level,
        )
        if not exercise_data:
            print(f"[Session] ✗ Exercise generation returned None — aborting")
            return None

        exercise = ExerciseRecord(
            exercise_id=str(uuid.uuid4())[:8],
            concept_idx=concept_idx,
            concept_name=concept_name,
            bloom_level=bloom_level,
            question=exercise_data["question"],
            options=exercise_data.get("options", {}),
            correct_option=exercise_data.get("correct_option", "A"),
            explanation=exercise_data.get("explanation", ""),
            # Theory will be handled directly in the frontend component mapping now,
            # or we can attach an empty structure.
            theory=None, 
        )

        session.current_exercise = exercise
        return exercise

    async def submit_answer(self, session_id: str, answer: str) -> Optional[Dict[str, Any]]:
        """Process user's answer, update environment, return result."""
        session = self._sessions.get(session_id)
        if not session or not session.current_exercise:
            return None

        exercise = session.current_exercise
        is_correct = answer.strip().upper() == exercise.correct_option.strip().upper()
        verdict = "✓ ĐÚNG" if is_correct else "✗ SAI"

        print(f"\n{'═'*60}")
        print(f"[Session] 📝 submit_answer")
        print(f"  Concept  : {exercise.concept_name} (idx={exercise.concept_idx})")
        print(f"  Answer   : {answer} → Correct: {exercise.correct_option} → {verdict}")

        # Evaluate with LLM (run in thread to avoid blocking event loop)
        loop = asyncio.get_event_loop()
        eval_result = await loop.run_in_executor(
            self._llm_executor,
            evaluate_answer,
            exercise.question, answer, exercise.correct_option, exercise.concept_name,
        )

        # Update exercise record
        exercise.user_answer = answer
        exercise.is_correct = is_correct
        session.exercise_history.append(exercise)

        if is_correct:
            session.total_correct += 1
        session.total_answered += 1

        # Step environment with human answer
        action_id = getattr(session, "_pending_action", 0)
        obs, reward, terminated, truncated, info = session.env.step(action_id, human_correct=is_correct)
        session.current_obs = obs
        session.current_exercise = None

        if terminated:
            session.status = "completed"

        # Get updated mastery
        mastery_matrix = session.env.get_mastery_matrix()
        concept_mastery = session.env.get_concept_mastery()
        mastery_val = float(concept_mastery[exercise.concept_idx])
        avg_mastery  = float(np.mean(concept_mastery))

        print(f"[Session] Mastery after: {mastery_val:.3f} | Avg: {avg_mastery:.3f} | Reward: {reward:.3f}")
        print(f"[Session] Step  : {info['step']} | Session status: {session.status}")
        print(f"{'═'*60}")

        # Persist session to MongoDB (fire-and-forget, non-blocking)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(mongo_store.save_session(session))
        except Exception as _mongo_err:
            print(f"[MongoDB] save_session schedule error: {_mongo_err}")

        return {
            "is_correct": is_correct,
            "explanation": eval_result["explanation"],
            "correct_option": exercise.correct_option,
            "concept_name": exercise.concept_name,
            "bloom_level": exercise.bloom_level,
            "mastery_after": mastery_val,
            "avg_mastery": avg_mastery,
            "step": info["step"],
            "session_completed": session.status == "completed",
            "stats": {
                "total_correct": session.total_correct,
                "total_answered": session.total_answered,
                "accuracy": session.total_correct / max(session.total_answered, 1),
            },
        }

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get full session status."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        env_stats = session.env.get_session_stats()
        concept_mastery = session.env.get_concept_mastery()
        id_to_concept = {v: k for k, v in session.concept_map.items()}

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

    # Sentence-transformer model for encoding new concepts (lazy-loaded)
    _text_encoder = None
    _TEXT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

    @classmethod
    def _get_text_encoder(cls):
        """Lazy-load sentence-transformer (same model used in SAINT training)."""
        if cls._text_encoder is None:
            from sentence_transformers import SentenceTransformer
            print(f"[Session] Loading text encoder: {cls._TEXT_MODEL_NAME}")
            cls._text_encoder = SentenceTransformer(cls._TEXT_MODEL_NAME)
            print("[Session] ✓ Text encoder ready")
        return cls._text_encoder

    def _encode_concepts(self, concept_names_ordered: List[str]) -> np.ndarray:
        """Encode concept names into 768d embeddings using sentence-transformers."""
        encoder = self._get_text_encoder()
        embeddings = encoder.encode(concept_names_ordered, show_progress_bar=False, batch_size=32)
        return np.array(embeddings, dtype=np.float32)

    async def create_session_from_pipeline(
        self,
        concepts_data: Dict[str, Any],
        concept_map: Dict[str, int],
        prereq_edges: List[Dict],
        max_steps: int = 50,
        precomputed_embeddings: Optional[List[List[float]]] = None,
        job_id: Optional[str] = None,
    ) -> SessionState:
        """Create a learning session from PDF pipeline output.

        Generates proper sentence-transformer embeddings for new concepts
        so SAINT can make meaningful predictions. If precomputed_embeddings
        are provided (from pipeline cache/MongoDB), uses those instead.
        
        If a previous session exists for this job_id in MongoDB, loads
        the exercise history so SAINT resumes from saved progress.
        """
        session_id = str(uuid.uuid4())[:8]

        # Build prereq graph for pipeline concepts
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
            print(
                f"[Session] create_session_from_pipeline: dropped {dropped_edges}/{len(prereq_edges)} "
                f"edges due to unresolved IDs"
            )

        # Build names/defs from pipeline
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

        # --- Build concept embeddings for SAINT ---
        n_pipeline = len(concept_map)
        if precomputed_embeddings is not None:
            print(f"[Session] Using precomputed embeddings ({len(precomputed_embeddings)} concepts)")
            raw_emb = np.array(precomputed_embeddings, dtype=np.float32)
        else:
            # Encode concept names using sentence-transformers
            ordered_names = []
            for idx in range(n_pipeline):
                cid = id_to_concept.get(idx, str(idx))
                name = names.get(cid, cid)
                definition = defs.get(cid, "")
                # Use "name: definition" for richer embedding
                text = f"{name}: {definition}" if definition else name
                ordered_names.append(text)
            print(f"[Session] Encoding {n_pipeline} new concepts with sentence-transformers...")
            raw_emb = self._encode_concepts(ordered_names)
            print(f"[Session] ✓ Embeddings shape: {raw_emb.shape}")

        # Build padded embedding tensor: row 0 = PAD (zeros), rows 1..K = concept embeddings
        emb_dim = raw_emb.shape[1]  # 768
        padded = np.zeros((n_pipeline + 1, emb_dim), dtype=np.float32)
        padded[1:n_pipeline + 1] = raw_emb
        external_embeddings = torch.from_numpy(padded).to(self._device)

        # Create env with external embeddings
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

        # --- Load previous history from MongoDB ---
        prev_history = []
        total_correct = 0
        total_answered = 0
        if job_id:
            try:
                prev_session = await mongo_store.find_latest_session_for_job(job_id)
                if prev_session and prev_session.get("exercise_history"):
                    prev_history = prev_session["exercise_history"]
                    total_correct = prev_session.get("total_correct", 0)
                    total_answered = prev_session.get("total_answered", 0)
                    print(f"[Session] Found previous session with {len(prev_history)} exercises ")
                    print(f"[Session]   Accuracy: {total_correct}/{total_answered} "
                          f"({total_correct/max(total_answered,1)*100:.0f}%)")
                else:
                    print(f"[Session] No previous session found for job {job_id}, starting fresh")
            except Exception as e:
                print(f"[Session] Error loading previous session: {e}, starting fresh")

        # Inject history into environment
        if prev_history:
            concept_indices = [ex["concept_idx"] for ex in prev_history]
            bloom_levels = [ex["bloom_level"] for ex in prev_history]
            responses = [1 if ex.get("is_correct") else 0 for ex in prev_history]
            env.inject_history(concept_indices, bloom_levels, responses)
            obs = env._build_obs()  # rebuild observation after injection

        session = SessionState(
            session_id=session_id,
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
                user_answer=ex.get("user_answer"),
                is_correct=ex.get("is_correct"),
                timestamp=ex.get("timestamp", 0),
            ))

        self._sessions[session_id] = session
        return session

    @property
    def concept_map(self):
        return self._concept_map

    @property
    def concept_names(self):
        return self._concept_names

    @property
    def n_concepts(self):
        return len(self._concept_map)
