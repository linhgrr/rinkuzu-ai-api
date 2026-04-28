"""Extraction chain for concept extraction from chunks."""

import os
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

from . import get_llm
from .schemas import ConceptExtraction, EvidenceVerification
from ..prompts import EXTRACTION_PROMPT, EVIDENCE_VERIFICATION_PROMPT
from ..utils.timeit import timeit
from loguru import logger
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

class ExtractionChain:
    """Chain for extracting concepts from text chunks using exact structured outputs."""

    def __init__(self, client=None):
        """
        Initialize extraction chain.

        Args:
            client: Optional ChatOpenAI instance
        """
        if client is None:
            self.llm = get_llm(
                temperature=0.1,  # A bit of temperature can sometimes help reasoning
                max_tokens=None,
                timeout=150,
                top_p=1,
            )
        else:
            self.llm = client

        # ---------------------------------------------------------
        # Extraction Prompt & Chain
        # ---------------------------------------------------------
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(EXTRACTION_PROMPT),
            HumanMessagePromptTemplate.from_template(
                "## DOCUMENT INFO\n\n* **subject_id**: {subject_id}\n\n"
                "{previous_concepts_section}"
                "## TEXT TO ANALYZE\n\n{text_content}"
            )
        ])

        try:
            self.structured_extract_llm = self.llm.with_structured_output(
                ConceptExtraction, method="json_mode"
            )
            self.extraction_chain = self.extraction_prompt | self.structured_extract_llm
        except Exception as e:
            logger.error(f"Failed to bind structured output for extraction: {e}")
            raise

        # ---------------------------------------------------------
        # Verification Prompt & Chain
        # ---------------------------------------------------------
        self.verification_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(EVIDENCE_VERIFICATION_PROMPT),
            HumanMessagePromptTemplate.from_template(
                "## CONCEPTS TO ANALYZE:\n\n**Concept A:** {concept_a}\n**Concept B:** {concept_b}"
            )
        ])

        try:
            self.structured_verif_llm = self.llm.with_structured_output(
                EvidenceVerification, method="json_mode"
            )
            self.verification_chain = self.verification_prompt | self.structured_verif_llm
        except Exception as e:
            logger.error(f"Failed to bind structured output for verification: {e}")
            raise

        logger.info("ExtractionChain initialized successfully using Native Structured Output.")

    @timeit
    def extract_from_batch(
        self,
        chunks: List[str],
        subject_id: str,
        batch_size: int = 5,
        max_workers: int = 8,
        max_previous_concepts: int = 20,
    ) -> List[ConceptExtraction]:
        """
        Extract concepts from multiple chunks with sequential batching.
        """
        batches = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batches.append(batch_chunks)

        logger.info(
            f"Processing {len(chunks)} chunks in {len(batches)} batches "
            f"(sequential mode, max_previous_concepts={max_previous_concepts})"
        )

        results: List[ConceptExtraction] = []
        all_previous_concepts: List[tuple] = []  # [(concept_id, name), ...]

        for batch_idx, batch in enumerate(batches):
            context_window = all_previous_concepts[-max_previous_concepts:] if max_previous_concepts > 0 else []

            try:
                result = self._process_batch(
                    batch,
                    subject_id,
                    batch_idx,
                    previous_concepts=context_window,
                )
                results.append(result)

                # Keep accumulating previously extracted concepts to enforce constraint correctness
                for concept in result.concepts:
                    entry = (concept.concept_id, concept.name)
                    if entry not in all_previous_concepts:
                        all_previous_concepts.append(entry)
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)[:100]}")
                results.append(
                    ConceptExtraction(
                        concepts=[],
                        subject_id=subject_id,
                        notes=f"Error: {str(e)[:100]}",
                    )
                )

        return results
    
    def _process_batch(
        self,
        batch_chunks: List[str],
        subject_id: str,
        batch_idx: int,
        previous_concepts: List[tuple] | None = None,
    ) -> ConceptExtraction:
        """
        Process a single batch of chunks relying SOLELY on native structured output.
        """
        if previous_concepts is None:
            previous_concepts = []

        try:
            combined_text = "\n".join(batch_chunks)

            if previous_concepts:
                names_formatted = "\n".join(
                    f"  - `{cid}` : {name}" for cid, name in previous_concepts
                )
                previous_concepts_section = (
                    "## CÁC KHÁI NIỆM ĐÃ TRÍCH XUẤT (từ các batch trước)\n\n"
                    "Danh sách các khái niệm đã được trích xuất ở các batch trước đây "
                    "(format: `concept_id` : tên khái niệm).\n"
                    "**QUAN TRỌNG**: Khi tạo relation với các khái niệm này, phải dùng ĐÚNG `concept_id` "
                    "(phần trước dấu `:`) làm `target_id`.\n\n"
                    f"{names_formatted}\n\n"
                )
            else:
                previous_concepts_section = ""

            logger.info(f"Batch {batch_idx} LLM Input (first 500 chars): {combined_text[:500]}...")

            invoke_args = {
                "subject_id": subject_id,
                "text_content": combined_text,
                "previous_concepts_section": previous_concepts_section,
            }

            # Only do native API call, no regex JSON repairing required via `json_mode`
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    result = self.extraction_chain.invoke(invoke_args)
                    
                    if not isinstance(result, ConceptExtraction):
                        raise ValueError(f"LLM API did not return a ConceptExtraction instance. Got: {type(result)}")
                        
                    logger.info(f"Batch {batch_idx} extraction success: {len(result.concepts)} concepts")
                    return result
                except Exception as e:
                    last_error = e
                    logger.warning(f"Batch {batch_idx} (Attempt {attempt+1}/{max_retries}) failed: {e}")
            
            logger.error(f"Batch {batch_idx} exhausted max_retries ({max_retries}). Last error: {last_error}")
            return ConceptExtraction(
                concepts=[], 
                subject_id=subject_id, 
                notes=f"Structured output failed after {max_retries} attempts. Error: {str(last_error)[:100]}"
            )

        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)[:100]}")
            return ConceptExtraction(concepts=[], subject_id=subject_id, notes=f"Error: {str(e)[:100]}")

    @timeit
    def verify_relation(
        self,
        concept_a: str,
        concept_b: str,
    ) -> EvidenceVerification:
        """
        Verify if a relation exists between two concepts using agent with tools.
        """
        return self._verify_single_relation(
            concept_a=concept_a,
            concept_b=concept_b,
            pair_idx=-1,
            max_retries=3,
        )
    
    @timeit
    def verify_relations_batch(
        self,
        concept_pairs: List[Tuple[str, str]],
        max_workers: int = 8,
    ) -> List[EvidenceVerification]:
        """
        Verify relations for multiple concept pairs in parallel.
        """
        logger.info(f"Verifying {len(concept_pairs)} concept pairs with {max_workers} workers")
        
        results = [None] * len(concept_pairs)  
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            for idx, (concept_a, concept_b) in enumerate(concept_pairs):
                future = executor.submit(
                    self._verify_single_relation,
                    concept_a,
                    concept_b,
                    idx
                )
                future_to_index[future] = idx
            
            for future in future_to_index:
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Error verifying pair {idx}: {str(e)[:100]}")
                    concept_a, concept_b = concept_pairs[idx]
                    results[idx] = EvidenceVerification(
                        has_relation=False,
                        relation_type=None,
                        direction=None,
                        confidence=0.0,
                        evidences=[],
                        reasoning=f"Error during verification: {str(e)[:100]}"
                    )
        
        return results

    def _verify_single_relation(
        self,
        concept_a: str,
        concept_b: str,
        pair_idx: int,
        max_retries: int = 3,
    ) -> EvidenceVerification:
        """
        Verify a single relation utilizing strict JSON schema Native outputs.
        """
        pair_label = "single" if pair_idx < 0 else f"pair {pair_idx}"
        logger.debug(f"Verifying {pair_label}: '{concept_a}' <-> '{concept_b}'")

        invoke_args = {"concept_a": concept_a, "concept_b": concept_b}

        last_error = None
        for attempt in range(max_retries):
            try:
                result = self.verification_chain.invoke(invoke_args)
                
                if not isinstance(result, EvidenceVerification):
                    raise ValueError(f"LLM API did not return EvidenceVerification instance. Got: {type(result)}")
                    
                logger.info(
                    f"Verified {pair_label}: {concept_a} <-> {concept_b} "
                    f"(has_relation={result.has_relation})"
                )
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    f"{pair_label} attempt {attempt+1}/{max_retries} failed: {str(e)[:120]}"
                )

        logger.error(
            f"{pair_label} ('{concept_a}' <-> '{concept_b}'): "
            f"failed after {max_retries} attempts"
        )
        return EvidenceVerification(
            has_relation=False,
            relation_type=None,
            direction=None,
            confidence=0.0,
            evidences=[],
            reasoning=f"Failed to generate structured output after {max_retries} attempts. Last error: {str(last_error)[:100]}"
        )
