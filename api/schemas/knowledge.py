"""
schemas/knowledge.py — Knowledge graph and mastery Pydantic models.
"""

from typing import List

from pydantic import BaseModel


class KnowledgeNodeResponse(BaseModel):
    id: str
    index: int
    name: str
    mastery: float
    status: str
    visited: bool


class KnowledgeEdgeResponse(BaseModel):
    source: str
    target: str


class KnowledgeGraphResponse(BaseModel):
    nodes: List[KnowledgeNodeResponse]
    edges: List[KnowledgeEdgeResponse]


class MasteryRow(BaseModel):
    concept_id: str
    concept_name: str
    bloom_levels: List[float]


class MasteryMatrixResponse(BaseModel):
    matrix: List[MasteryRow]
    bloom_labels: List[str]


class ConceptPrereq(BaseModel):
    id: str
    name: str
    mastery: float


class ConceptDetailResponse(BaseModel):
    id: str
    name: str
    definition: str
    mastery: float
    status: str
    bloom_mastery: List[float]
    prerequisites: List[ConceptPrereq]
    dependents: List[ConceptPrereq]
    visited: bool
    visit_count: int
