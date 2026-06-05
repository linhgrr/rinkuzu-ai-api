# Pydantic schemas for concept extraction / knowledge graph.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSchemaModel(BaseModel):
    """Base model for provider-agnostic structured LLM outputs."""

    model_config = ConfigDict(extra="forbid")


class Relation(StrictSchemaModel):
    """Dependency relation attached to the current extracted concept."""

    type: Literal["PREREQUISITE"] = Field(
        ...,
        description=("Relation type. PREREQUISITE means the current concept depends on target_id."),
    )
    target_id: str = Field(
        ...,
        min_length=1,
        description="Concept ID of the prerequisite required by the current concept.",
    )
    confidence: float | None = Field(
        None,
        ge=0,
        le=1,
        description="Confidence score for the relation (recommended to save if >= threshold).",
    )
    evidence: str | None = Field(
        None, description="Evidence text supporting the relation (can be empty for inferred)."
    )


class Formula(StrictSchemaModel):
    """Structured formula attached to a concept."""

    latex: str = Field(..., description="LaTeX expression of the formula.")
    description: str | None = Field(
        None, description="Formula explanation (meaning, application conditions)."
    )
    variables: dict[str, str] | None = Field(
        None, description="Variable glossary, e.g., {'I': 'current (A)', 'R': 'resistance (Ω)'}."
    )


class Concept(StrictSchemaModel):
    """Atomic educational concept within a subject-level knowledge graph."""

    concept_id: str = Field(..., min_length=1, description="Unique identifier for the concept.")
    subject_id: str = Field(
        ..., min_length=1, description="ID of the subject/course that the concept belongs to."
    )
    name: str = Field(..., min_length=1, description="Primary name of ONE concept.")
    definition: str = Field(
        ..., min_length=1, description="Concise, clear definition. Must provide"
    )
    examples: list[str] = Field(
        default_factory=list, description="Illustrative examples for the concept."
    )
    formulas: list[Formula] = Field(
        default_factory=list, description="List of related formulas (if any)."
    )
    relations: list[Relation] = Field(
        default_factory=list,
        description=(
            "Prerequisites required by this concept. The graph converts each relation "
            "to target_id -> current concept."
        ),
    )

    name_embedding: list[float] | None = Field(
        None,
        description=(
            "DEPRECATED. Vietnamese-sbert embedding of the concept name. Kept "
            "optional for backward compatibility with existing MongoDB documents; "
            "no longer populated by the live pipeline (MLPPrerequisiteRanker "
            "carries its own XLM-RoBERTa encoder)."
        ),
    )
    definition_embedding: list[float] | None = Field(
        None,
        description=(
            "DEPRECATED. Vietnamese-sbert embedding of the definition. Kept "
            "optional for backward compatibility; no longer populated."
        ),
    )


class ConceptExtraction(StrictSchemaModel):
    """Model output for a batch of extracted concepts."""

    concepts: list[Concept] = Field(..., description="List of extracted concepts.")
    subject_id: str | None = Field(
        None, description="Subject applied to the entire batch (if any)."
    )
    notes: str | None = Field(
        None, description="Notes (warnings, statistics, heuristic decisions, etc.)."
    )


class ExtractionConceptPayload(StrictSchemaModel):
    """LLM tool-call payload for a single extracted concept."""

    concept_id: str = Field(..., min_length=1, description="Unique identifier for the concept.")
    subject_id: str = Field(
        ..., min_length=1, description="ID of the subject/course that the concept belongs to."
    )
    name: str = Field(..., min_length=1, description="Primary Vietnamese name of ONE concept.")
    definition: str = Field(..., min_length=1, description="Concise Vietnamese definition.")
    examples: list[str] = Field(
        default_factory=list, description="Illustrative Vietnamese examples for the concept."
    )
    formulas: list[Formula] = Field(
        default_factory=list, description="List of related formulas (if any)."
    )
    relations: list[Relation] = Field(
        default_factory=list,
        description=(
            "Prerequisites required by this concept. For PREREQUISITE, target_id is "
            "the prerequisite concept id, not the dependent concept id."
        ),
    )


class ConceptExtractionPayload(StrictSchemaModel):
    """LLM tool-call payload for one PDF batch."""

    concepts: list[ExtractionConceptPayload] = Field(..., description="List of extracted concepts.")
    subject_id: str | None = Field(
        None, description="Subject applied to the entire batch (if any)."
    )
    notes: str | None = Field(
        None, description="Warnings, observations, or extraction notes for this batch."
    )


def materialize_concept_extraction(payload: ConceptExtractionPayload) -> ConceptExtraction:
    """Convert lightweight LLM payload into the domain model used by the pipeline."""

    return ConceptExtraction(
        concepts=[
            Concept(
                concept_id=concept.concept_id,
                subject_id=concept.subject_id,
                name=concept.name,
                definition=concept.definition,
                examples=list(concept.examples),
                formulas=concept.formulas,
                relations=list(concept.relations),
                name_embedding=None,
                definition_embedding=None,
            )
            for concept in payload.concepts
        ],
        subject_id=payload.subject_id,
        notes=payload.notes,
    )


class EvidenceVerification(StrictSchemaModel):
    """Result of verifying a relation between two concepts."""

    has_relation: bool = Field(
        ..., description="Whether a relation exists between the two concepts."
    )
    relation_type: Literal["PREREQUISITE", "SAME_CONCEPT"] | None = Field(
        None,
        description="Type of relation if one exists. SAME_CONCEPT means the two concepts are identical and should be merged.",
    )
    direction: Literal["A_to_B", "B_to_A", "same_concept"] | None = Field(
        None,
        description="Direction of the relation. A_to_B means concept_a is prerequisite/related to concept_b. same_concept means they are the same concept and should be merged.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the verification (0.0 to 1.0)."
    )
    evidences: list[str] = Field(
        default_factory=list, description="Text evidences supporting the relation."
    )
    reasoning: str | None = Field(
        None, description="Explanation of why this relation exists or doesn't exist."
    )


class EdgeDecision(StrictSchemaModel):
    """Decision for a single edge in a cycle."""

    source_id: str = Field(..., description="Source concept ID of the edge")
    target_id: str = Field(..., description="Target concept ID of the edge")
    should_remove: bool = Field(
        ..., description="Whether this edge should be removed to break the cycle"
    )
    reasoning: str = Field(
        ..., description="Explanation of why this edge should or should not be removed"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in this decision (0.0 to 1.0)"
    )


class CycleRemovalDecision(StrictSchemaModel):
    """LLM decision for removing edges from a cycle."""

    cycle_nodes: list[str] = Field(..., description="List of concept IDs forming the cycle")
    edges_to_remove: list[EdgeDecision] = Field(
        ..., description="Edges that should be removed to break the cycle"
    )
    reasoning: str = Field(..., description="Overall reasoning for the cycle removal strategy")
