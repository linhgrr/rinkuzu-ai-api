"""
Evaluation script for prerequisite ranking on Lecture Bank dataset.

This script evaluates the prerequisite ranking pipeline on the Lecture Bank dataset,
which contains 208 NLP/ML concepts with prerequisite annotations.

Usage:
    python eval_lecture_bank.py [--prs-threshold 0.75] [--min-confidence 0.5] [--output results.json]
"""
from loguru import logger
from embed.prereq_ranking import rank_prerequisites
from embed.embeddings import compute_embedding_for_concepts
from embed.embedding_client import EmbeddingClient
from llm.schemas import Concept, Relation
from llm.extract_chain import ExtractionChain
from llm import get_llm
from config import settings as config_settings
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import numpy as np
from typing import List, Dict, Tuple, Set
import csv
import json
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class LectureBankEvaluator:
    """Evaluator for Lecture Bank dataset."""

    def __init__(
        self,
        dataset_dir: str = "dataset/lecture_bank",
        prs_threshold: float = None,
        min_confidence: float = 0.5,
        use_verification: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            dataset_dir: Path to Lecture Bank dataset
            prs_threshold: PRS threshold for ranking (default from settings)
            min_confidence: Minimum confidence for verified relations
            use_verification: Whether to use LLM verification
        """
        self.dataset_dir = Path(dataset_dir)
        self.prs_threshold = prs_threshold or config_settings.prs_threshold
        self.min_confidence = min_confidence
        self.use_verification = use_verification

        # Load dataset
        self.concepts = self._load_concepts()
        self.ground_truth = self._load_ground_truth()

        # Initialize models
        logger.info("Initializing models...")
        self.embedding_client = EmbeddingClient()

        if self.use_verification:
            llm = get_llm(
                temperature=0.1,
            )
            self.extraction_chain = ExtractionChain(client=llm)
        else:
            self.extraction_chain = None

        logger.info(
            f"Loaded {len(self.concepts)} concepts, "
            f"{len(self.ground_truth)} ground truth relations"
        )

    def _load_concepts(self) -> List[Concept]:
        """Load concepts from 208topics_with_definitions.csv."""
        concepts = []
        csv_path = self.dataset_dir / "208topics_with_definitions.csv"

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue

                concept_id = row[0].strip()
                name = row[1].strip()
                url = row[2].strip()
                definition = row[3].strip()

                concept = Concept(
                    concept_id=concept_id,
                    subject_id="lecture_bank",
                    name=name,
                    definition=definition,
                    examples=[],
                    relations=[]
                )
                concepts.append(concept)

        return concepts

    def _load_ground_truth(self) -> Dict[Tuple[str, str], int]:
        """
        Load ground truth from prerequisite_annotation.csv.

        Returns:
            Dict mapping (concept_id_1, concept_id_2) -> label
            where label = 1 means concept_id_1 is prerequisite of concept_id_2
        """
        ground_truth = {}
        csv_path = self.dataset_dir / "prerequisite_annotation.csv"

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue

                id1 = row[0].strip()
                id2 = row[1].strip()
                label = int(row[2].strip())

                ground_truth[(id1, id2)] = label

        return ground_truth

    def run_evaluation(self) -> Dict:
        """
        Run full evaluation pipeline.

        Returns:
            Dictionary with evaluation results
        """
        results = {
            'config': {
                'prs_threshold': self.prs_threshold,
                'min_confidence': self.min_confidence,
                'use_verification': self.use_verification,
                'num_concepts': len(self.concepts),
                'num_ground_truth': len(self.ground_truth),
            },
            'stages': {}
        }

        # Stage 1: Generate embeddings
        logger.info("Stage 1: Generating embeddings...")
        compute_embedding_for_concepts(
            concepts=self.concepts,
            client=self.embedding_client
        )

        # Stage 2: Rank prerequisites
        logger.info(
            f"Stage 2: Ranking prerequisites (threshold={self.prs_threshold})...")
        prereq_pairs = rank_prerequisites(
            concepts=self.concepts,
            prs_threshold=self.prs_threshold
        )

        logger.info(f"Found {len(prereq_pairs)} candidate pairs")

        # Evaluate ranking stage
        ranking_metrics = self._evaluate_ranking(prereq_pairs)
        results['stages']['ranking'] = ranking_metrics

        # Stage 3: Verification (optional)
        if self.use_verification and prereq_pairs:
            logger.info(f"Stage 3: Verifying {len(prereq_pairs)} pairs...")
            verified_pairs = self._verify_pairs(prereq_pairs)

            logger.info(f"Verified {len(verified_pairs)} pairs")

            # Evaluate verification stage
            verification_metrics = self._evaluate_verification(verified_pairs)
            results['stages']['verification'] = verification_metrics

            # Combined metrics (ranking + verification)
            combined_metrics = self._evaluate_combined(
                prereq_pairs, verified_pairs)
            results['combined'] = combined_metrics

        return results

    def _verify_pairs(self, prereq_pairs: List[Tuple[str, str]]) -> List[Dict]:
        """
        Verify prerequisite pairs using LLM.

        Args:
            prereq_pairs: List of (concept_id_1, concept_id_2) tuples

        Returns:
            List of verified relations with metadata
        """
        concept_map = {c.concept_id: c for c in self.concepts}

        # Prepare pairs for verification
        concept_pairs = []
        pair_metadata = []

        for id1, id2 in tqdm(prereq_pairs, desc="Preparing pairs"):
            c1 = concept_map.get(id1)
            c2 = concept_map.get(id2)

            if c1 and c2:
                concept_pairs.append((c1.name, c2.name))
                pair_metadata.append({
                    'concept_a_id': id1,
                    'concept_b_id': id2,
                    'concept_a_name': c1.name,
                    'concept_b_name': c2.name
                })

        # Verify in parallel
        logger.info(f"Verifying {len(concept_pairs)} pairs with LLM...")
        verifications = self.extraction_chain.verify_relations_batch(
            concept_pairs=concept_pairs,
            max_workers=4
        )

        # Combine results
        verified_pairs = []
        for metadata, verification in zip(pair_metadata, verifications):
            if (verification.has_relation and
                verification.direction != "same_concept" and
                    verification.confidence >= self.min_confidence):

                # Determine actual direction
                if verification.direction == "A_to_B":
                    source_id = metadata['concept_a_id']
                    target_id = metadata['concept_b_id']
                elif verification.direction == "B_to_A":
                    source_id = metadata['concept_b_id']
                    target_id = metadata['concept_a_id']
                else:
                    continue

                verified_pairs.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'confidence': verification.confidence,
                    'relation_type': verification.relation_type,
                    'original_pair': (metadata['concept_a_id'], metadata['concept_b_id'])
                })

        return verified_pairs

    def _evaluate_ranking(self, prereq_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Evaluate ranking stage against ground truth.

        Note: Ranking is undirected (doesn't predict direction),
        so we check if either (A,B) or (B,A) is in ground truth.
        """
        predicted_set = set(prereq_pairs)

        # Build sets for evaluation
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        # Check all possible pairs
        concept_ids = [c.concept_id for c in self.concepts]
        n = len(concept_ids)

        for i in range(n):
            for j in range(i + 1, n):
                id1, id2 = concept_ids[i], concept_ids[j]

                # Check if predicted
                is_predicted = (id1, id2) in predicted_set

                # Check ground truth (either direction)
                has_relation = (
                    self.ground_truth.get((id1, id2), 0) == 1 or
                    self.ground_truth.get((id2, id1), 0) == 1
                )

                if is_predicted and has_relation:
                    true_positives += 1
                elif is_predicted and not has_relation:
                    false_positives += 1
                elif not is_predicted and has_relation:
                    false_negatives += 1
                else:
                    true_negatives += 1

        # Calculate metrics
        precision = true_positives / \
            (true_positives + false_positives) if (true_positives +
                                                   false_positives) > 0 else 0
        recall = true_positives / \
            (true_positives + false_negatives) if (true_positives +
                                                   false_negatives) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        return {
            'num_predicted': len(prereq_pairs),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def _evaluate_verification(self, verified_pairs: List[Dict]) -> Dict:
        """
        Evaluate verification stage (with direction).

        This checks if the predicted direction matches ground truth.
        """
        # Build predicted set with direction
        predicted_directed = {(p['source_id'], p['target_id'])
                              for p in verified_pairs}

        # Check against ground truth
        true_positives = 0
        false_positives = 0

        for source_id, target_id in predicted_directed:
            if self.ground_truth.get((source_id, target_id), 0) == 1:
                true_positives += 1
            else:
                false_positives += 1

        # Count false negatives (ground truth relations not predicted)
        false_negatives = 0
        for (id1, id2), label in self.ground_truth.items():
            if label == 1 and (id1, id2) not in predicted_directed:
                false_negatives += 1

        # Calculate metrics
        precision = true_positives / \
            (true_positives + false_positives) if (true_positives +
                                                   false_positives) > 0 else 0
        recall = true_positives / \
            (true_positives + false_negatives) if (true_positives +
                                                   false_negatives) > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        return {
            'num_verified': len(verified_pairs),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def _evaluate_combined(
        self,
        prereq_pairs: List[Tuple[str, str]],
        verified_pairs: List[Dict]
    ) -> Dict:
        """
        Evaluate combined pipeline (ranking + verification).

        This measures how many ground truth relations were:
        1. Found by ranking
        2. Correctly verified with direction
        """
        # Pairs found by ranking
        ranking_found = set(prereq_pairs)

        # Pairs verified with direction
        verified_directed = {(p['source_id'], p['target_id'])
                             for p in verified_pairs}

        # Ground truth relations
        gt_relations = {(id1, id2) for (id1, id2),
                        label in self.ground_truth.items() if label == 1}

        # Analyze pipeline stages
        gt_total = len(gt_relations)

        # How many GT relations were found by ranking?
        ranking_recall = 0
        for id1, id2 in gt_relations:
            if (id1, id2) in ranking_found or (id2, id1) in ranking_found:
                ranking_recall += 1

        # How many GT relations were correctly verified?
        verification_recall = 0
        for id1, id2 in gt_relations:
            if (id1, id2) in verified_directed:
                verification_recall += 1

        return {
            'ground_truth_total': gt_total,
            'found_by_ranking': ranking_recall,
            'correctly_verified': verification_recall,
            'ranking_recall': ranking_recall / gt_total if gt_total > 0 else 0,
            'verification_recall': verification_recall / gt_total if gt_total > 0 else 0,
            'pipeline_recall': verification_recall / gt_total if gt_total > 0 else 0,
        }

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

    def print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("LECTURE BANK EVALUATION RESULTS")
        print("=" * 80)

        # Config
        print("\n📊 Configuration:")
        print(f"  • PRS Threshold: {results['config']['prs_threshold']:.2f}")
        print(f"  • Min Confidence: {results['config']['min_confidence']:.2f}")
        print(f"  • Use Verification: {results['config']['use_verification']}")
        print(f"  • Concepts: {results['config']['num_concepts']}")
        print(
            f"  • Ground Truth Relations: {results['config']['num_ground_truth']}")

        # Ranking stage
        print("\n🔗 Stage 1: Prerequisite Ranking (PRS)")
        ranking = results['stages']['ranking']
        print(f"  • Candidates Found: {ranking['num_predicted']}")
        print(f"  • Precision: {ranking['precision']:.4f}")
        print(f"  • Recall: {ranking['recall']:.4f}")
        print(f"  • F1 Score: {ranking['f1']:.4f}")

        # Verification stage
        if 'verification' in results['stages']:
            print("\n🔍 Stage 2: LLM Verification")
            verification = results['stages']['verification']
            print(f"  • Verified: {verification['num_verified']}")
            print(f"  • Precision: {verification['precision']:.4f}")
            print(f"  • Recall: {verification['recall']:.4f}")
            print(f"  • F1 Score: {verification['f1']:.4f}")

        # Combined
        if 'combined' in results:
            print("\n🎯 Combined Pipeline Performance")
            combined = results['combined']
            print(f"  • Ground Truth Total: {combined['ground_truth_total']}")
            print(
                f"  • Found by Ranking: {combined['found_by_ranking']} ({combined['ranking_recall']:.2%})")
            print(
                f"  • Correctly Verified: {combined['correctly_verified']} ({combined['verification_recall']:.2%})")
            print(
                f"  • Overall Pipeline Recall: {combined['pipeline_recall']:.2%}")

        print("\n" + "=" * 80 + "\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate prerequisite ranking on Lecture Bank dataset"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset/lecture_bank',
        help='Path to Lecture Bank dataset directory'
    )
    parser.add_argument(
        '--prs-threshold',
        type=float,
        default=None,
        help='PRS threshold for ranking (default from settings)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence for verification (default: 0.5)'
    )
    parser.add_argument(
        '--no-verification',
        action='store_true',
        help='Skip LLM verification stage'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='eval_results.json',
        help='Output file for results (default: eval_results.json)'
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = LectureBankEvaluator(
        dataset_dir=args.dataset_dir,
        prs_threshold=args.prs_threshold,
        min_confidence=args.min_confidence,
        use_verification=not args.no_verification
    )

    # Run evaluation
    results = evaluator.run_evaluation()

    # Print summary
    evaluator.print_summary(results)

    # Save results
    evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
