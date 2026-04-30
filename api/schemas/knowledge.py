"""
schemas/knowledge.py — Knowledge graph and mastery Pydantic models.
"""


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
    nodes: list[KnowledgeNodeResponse]
    edges: list[KnowledgeEdgeResponse]


class MasteryRow(BaseModel):
    concept_id: str
    concept_name: str
    bloom_levels: list[float]


class MasteryMatrixResponse(BaseModel):
    matrix: list[MasteryRow]
    bloom_labels: list[str]


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
    bloom_mastery: list[float]
    prerequisites: list[ConceptPrereq]
    dependents: list[ConceptPrereq]
    visited: bool
    visit_count: int
