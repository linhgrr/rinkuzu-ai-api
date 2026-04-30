# Pydantic schemas for concept extraction / knowledge graph (Final)

from typing import Literal

from pydantic import BaseModel, Field


class Relation(BaseModel):
    """Directed relation from this concept to a target concept."""
    type: Literal["PREREQUISITE"] = Field(
        ...,
        description="Relation type. PREREQUISITE is a learning-oriented edge."
    )
    target_id: str = Field(
        ...,
        min_length=1,
        description="Target concept ID."
    )
    confidence: float | None = Field(
        None, ge=0, le=1,
        description="Confidence score for the relation (recommended to save if >= threshold)."
    )
    evidence: str | None = Field(
        None,
        description="Evidence text supporting the relation (can be empty for inferred)."
    )


class Formula(BaseModel):
    """Structured formula attached to a concept."""
    latex: str = Field(
        ...,
        description="LaTeX expression of the formula."
    )
    description: str | None = Field(
        None,
        description="Formula explanation (meaning, application conditions)."
    )
    variables: dict[str, str] | None = Field(
        None,
        description="Variable glossary, e.g., {'I': 'current (A)', 'R': 'resistance (Ω)'}."
    )


class Concept(BaseModel):
    """Atomic educational concept within a subject-level knowledge graph."""
    concept_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the concept."
    )
    subject_id: str = Field(
        ...,
        min_length=1,
        description="ID of the subject/course that the concept belongs to."
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Primary name of ONE concept."
    )
    definition: str = Field(
        ...,
        min_length=1,
        description="Concise, clear definition. Must provide"
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Illustrative examples for the concept."
    )
    formulas: list[Formula] | None = Field(
        None,
        description="List of related formulas (if any)."
    )
    relations: list[Relation] = Field(
        default_factory=list,
        description="Relations to other concepts (prioritize PREREQUISITE)."
    )

    name_embedding: list[float] | None = Field(
        None,
        description="Vector embedding of concept name only (for CSR prerequisite ranking)."
    )
    definition_embedding: list[float] | None = Field(
        None,
        description="Vector embedding of definition text only (for CSR prerequisite ranking)."
    )


class ConceptExtraction(BaseModel):
    """Model output for a batch of extracted concepts."""
    concepts: list[Concept] = Field(
        ...,
        description="List of extracted concepts."
    )
    subject_id: str | None = Field(
        None,
        description="Subject applied to the entire batch (if any)."
    )
    notes: str | None = Field(
        None,
        description="Notes (warnings, statistics, heuristic decisions, etc.)."
    )


class EvidenceVerification(BaseModel):
    """Result of verifying a relation between two concepts."""
    has_relation: bool = Field(
        ...,
        description="Whether a relation exists between the two concepts."
    )
    relation_type: Literal["PREREQUISITE", "SAME_CONCEPT"] | None = Field(
        None,
        description="Type of relation if one exists. SAME_CONCEPT means the two concepts are identical and should be merged."
    )
    direction: Literal["A_to_B", "B_to_A", "same_concept"] | None = Field(
        None,
        description="Direction of the relation. A_to_B means concept_a is prerequisite/related to concept_b. same_concept means they are the same concept and should be merged."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the verification (0.0 to 1.0)."
    )
    evidences: list[str] = Field(
        default_factory=list,
        description="Text evidences supporting the relation."
    )
    reasoning: str | None = Field(
        None,
        description="Explanation of why this relation exists or doesn't exist."
    )

class EdgeDecision(BaseModel):
    """Decision for a single edge in a cycle."""
    source_id: str = Field(
        ...,
        description="Source concept ID of the edge"
    )
    target_id: str = Field(
        ...,
        description="Target concept ID of the edge"
    )
    should_remove: bool = Field(
        ...,
        description="Whether this edge should be removed to break the cycle"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why this edge should or should not be removed"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this decision (0.0 to 1.0)"
    )


class CycleRemovalDecision(BaseModel):
    """LLM decision for removing edges from a cycle."""
    cycle_nodes: list[str] = Field(
        ...,
        description="List of concept IDs forming the cycle"
    )
    edges_to_remove: list[EdgeDecision] = Field(
        ...,
        description="Edges that should be removed to break the cycle"
    )
    reasoning: str = Field(
        ...,
        description="Overall reasoning for the cycle removal strategy"
    )
