"""Knowledge graph builder using NetworkX (DiGraph, no MultiGraph)."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from loguru import logger
import networkx as nx

if TYPE_CHECKING:
    from api.core.content_pipeline.infrastructure.llm.schemas import Concept


class RelationType(StrEnum):
    PREREQUISITE = "PREREQUISITE"
    UNKNOWN = "UNKNOWN"


class KnowledgeGraphBuilder:
    """Builder for knowledge graphs (DiGraph)."""

    def __init__(self, subject_id: str) -> None:
        """
        Initialize builder.

        Args:
            subject_id: Subject identifier
        """
        self.subject_id = subject_id
        self.graph = nx.DiGraph()
        self.concept_map: dict[str, Concept] = {}

    def _norm_rel(self, s: str | None) -> RelationType:
        val = (s or "").strip().upper()
        try:
            return RelationType[val]
        except Exception:
            return RelationType.UNKNOWN

    def _title_from_id(self, node_id: str) -> str:
        return " ".join(w.capitalize() for w in (node_id or "").split("_")).strip() or node_id

    def _ensure_node(self, node_id: str, *, placeholder: bool = True) -> Any:
        """Đảm bảo node tồn tại trong graph. Nếu chưa có thì tạo node placeholder."""
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                name=self._title_from_id(node_id),
                definition="",
                is_placeholder=placeholder,
            )

    def _set_non_placeholder(self, node_id: str, **attrs: Any) -> Any:
        """Khi add_concept thật, gỡ cờ placeholder và cập nhật thuộc tính."""
        self._ensure_node(node_id, placeholder=False)
        attrs = {"is_placeholder": False, **attrs}
        nx.set_node_attributes(self.graph, {node_id: attrs})

    def _normalize_evidence(self, evidence: str | list[str] | None) -> list[str] | None:
        """Đưa evidence về list[str] đã dedupe."""
        if evidence is None:
            return None

        items: list[str] = []
        items = evidence if isinstance(evidence, list) else [evidence]

        out: list[str] = []
        seen = set()
        for it in items:
            text = str(it).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out or None

    def _merge_evidence(
        self,
        old: str | list[str] | None,
        new: str | list[str] | None,
    ) -> list[str] | None:
        """Gộp evidence cũ & mới, trả về list[str] đã dedupe."""
        old_list = self._normalize_evidence(old) or []
        new_list = self._normalize_evidence(new) or []
        out: list[str] = []
        seen = set()
        for text in old_list + new_list:
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out or None

    def add_concepts(self, concepts: list[Concept]) -> Any:
        """Add multiple concepts to graph."""
        for concept in concepts:
            self.add_concept(concept)

    def add_concept(self, concept: Concept) -> Any:
        """Add a concept node.

        Relation extraction now produces verifier candidates. Edges are added
        only after verification by the pipeline graph-building stage.
        """
        self.add_concept_node(concept)

    def add_concept_nodes(self, concepts: list[Concept]) -> Any:
        """Add multiple concept nodes without relation side effects."""
        for concept in concepts:
            self.add_concept_node(concept)

    def add_concept_node(self, concept: Concept) -> Any:
        """Add a concept node without translating extracted relations into edges."""
        self._set_non_placeholder(
            concept.concept_id,
            name=getattr(concept, "name", None),
            definition=getattr(concept, "definition", None),
            examples=getattr(concept, "examples", None),
        )

        self.concept_map[concept.concept_id] = concept

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str | None = None,
        evidence: str | list[str] | None = None,
        location: str | None = None,
        confidence: float | None = None,
        reasoning: str | None = None,
        sources: list[str] | None = None,
        ranker_score: float | None = None,
        extraction_confidence: float | None = None,
    ) -> Any:
        """
        Add a relation (edge) to graph.

        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relation_type: Type of relation (str/enum-like)
            evidence: Evidence (str or list of str)
            location: Evidence location (optional)
        """
        self._ensure_node(source_id, placeholder=True)
        self._ensure_node(target_id, placeholder=True)

        rel_type = self._norm_rel(relation_type)

        old_data = self.graph.get_edge_data(source_id, target_id, default={})
        old_type = self._norm_rel(old_data.get("relation_type"))
        old_evd = old_data.get("evidence")

        if old_type not in (RelationType.UNKNOWN, rel_type) and old_data:
            logger.warning(
                "Overwriting relation_type {} -> {} for edge {}->{}",
                old_type,
                rel_type,
                source_id,
                target_id,
            )

        merged_evidence = self._merge_evidence(old_evd, evidence)

        self.graph.add_edge(
            source_id,
            target_id,
            relation_type=rel_type.value,
            evidence=merged_evidence,
            location=location,
            confidence=confidence,
            reasoning=reasoning,
            sources=sources or [],
            ranker_score=ranker_score,
            extraction_confidence=extraction_confidence,
        )

    def get_graph(self) -> nx.DiGraph:
        """Get the knowledge graph."""
        return self.graph

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics (kể cả kiểm tra cycle)."""
        edge_types: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation_type", RelationType.UNKNOWN.value)
            edge_types[rel] = edge_types.get(rel, 0) + 1

        has_cycle = not nx.is_directed_acyclic_graph(self.graph)

        return {
            "subject_id": self.subject_id,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "edge_types": edge_types,
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0.0,
            "has_cycle": has_cycle,
        }

    def get_prerequisites(self, concept_id: str) -> list[str]:
        """Get prerequisite concepts for a concept (incoming edges with PREREQUISITE)."""
        if concept_id not in self.graph:
            return []
        res: list[str] = []
        for pred in self.graph.predecessors(concept_id):
            data = self.graph[pred][concept_id]
            if self._norm_rel(data.get("relation_type")) == RelationType.PREREQUISITE:
                res.append(pred)
        return res

    def get_dependents(self, concept_id: str) -> list[str]:
        """Get concepts depending on this concept (outgoing PREREQUISITE edges)."""
        if concept_id not in self.graph:
            return []
        res: list[str] = []
        for succ in self.graph.successors(concept_id):
            data = self.graph[concept_id][succ]
            if self._norm_rel(data.get("relation_type")) == RelationType.PREREQUISITE:
                res.append(succ)
        return res
