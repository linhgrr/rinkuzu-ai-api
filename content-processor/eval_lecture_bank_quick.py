"""
Quick evaluation script for Lecture Bank dataset - No verification version.

This version only evaluates the PRS ranking stage without LLM verification,
making it much faster to run.

Usage:
    python eval_lecture_bank_quick.py --prs-threshold 0.75
"""
from loguru import logger
from embed.prereq_ranking import rank_prerequisites
from embed.embeddings import compute_embedding_for_concepts
from embed.embedding_client import EmbeddingClient
from llm.schemas import Concept
from config import settings as config_settings
from typing import List, Dict, Tuple
import csv
import json
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def load_concepts(dataset_dir: Path) -> List[Concept]:
    """Load concepts from 208topics_with_definitions.csv."""
    concepts = []
    csv_path = dataset_dir / "208topics_with_definitions.csv"

    logger.info(f"Loading concepts from {csv_path}")

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

    logger.info(f"Loaded {len(concepts)} concepts")
    return concepts


def load_ground_truth(dataset_dir: Path) -> Dict[Tuple[str, str], int]:
    """
    Load ground truth from prerequisite_annotation.csv.

    Returns:
        Dict mapping (concept_id_1, concept_id_2) -> label
        where label = 1 means concept_id_1 is prerequisite of concept_id_2
    """
    ground_truth = {}
    csv_path = dataset_dir / "prerequisite_annotation.csv"

    logger.info(f"Loading ground truth from {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue

            id1 = row[0].strip()
            id2 = row[1].strip()
            label = int(row[2].strip())

            ground_truth[(id1, id2)] = label

    # Count positive relations
    positive_count = sum(1 for label in ground_truth.values() if label == 1)
    logger.info(
        f"Loaded {len(ground_truth)} pairs ({positive_count} positive relations)")

    return ground_truth


def evaluate_ranking(
    prereq_pairs: List[Tuple[str, str]],
    ground_truth: Dict[Tuple[str, str], int],
    all_concept_ids: List[str]
) -> Dict:
    """
    Evaluate ranking against ground truth.

    Note: Ranking is undirected, so we check if either (A,B) or (B,A) is in ground truth.
    """
    predicted_set = set(prereq_pairs)

    # Build sets for evaluation
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    # Check all possible pairs
    n = len(all_concept_ids)
    total_pairs = n * (n - 1) // 2

    logger.info(
        f"Evaluating {len(prereq_pairs)} predictions against {total_pairs} possible pairs...")

    for i in range(n):
        for j in range(i + 1, n):
            id1, id2 = all_concept_ids[i], all_concept_ids[j]

            # Check if predicted
            is_predicted = (id1, id2) in predicted_set

            # Check ground truth (either direction)
            has_relation = (
                ground_truth.get((id1, id2), 0) == 1 or
                ground_truth.get((id2, id1), 0) == 1
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
    accuracy = (true_positives + true_negatives) / \
        total_pairs if total_pairs > 0 else 0

    return {
        'num_predicted': len(prereq_pairs),
        'total_possible_pairs': total_pairs,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
    }


def print_results(results: Dict):
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print("LECTURE BANK EVALUATION RESULTS (PRS Ranking Only)")
    print("=" * 80)

    # Config
    config = results['config']
    print(f"\n📊 Configuration:")
    print(f"  • PRS Threshold: {config['prs_threshold']:.3f}")
    print(f"  • Number of Concepts: {config['num_concepts']}")
    print(
        f"  • Ground Truth Relations: {config['num_positive_relations']}/{config['num_total_pairs']}")

    # Metrics
    metrics = results['metrics']
    print(f"\n🔗 Prerequisite Ranking Results:")
    print(
        f"  • Candidates Found: {metrics['num_predicted']}/{metrics['total_possible_pairs']}")
    print(f"  • True Positives: {metrics['true_positives']}")
    print(f"  • False Positives: {metrics['false_positives']}")
    print(f"  • False Negatives: {metrics['false_negatives']}")
    print(f"  • True Negatives: {metrics['true_negatives']}")
    print(f"\n📈 Performance Metrics:")
    print(
        f"  • Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(
        f"  • Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  • F1 Score: {metrics['f1']:.4f}")
    print(
        f"  • Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

    print("\n" + "=" * 80 + "\n")


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Quick evaluation of PRS ranking on Lecture Bank"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset/lecture_bank',
        help='Path to Lecture Bank dataset'
    )
    parser.add_argument(
        '--prs-threshold',
        type=float,
        default=None,
        help='PRS threshold (default from settings)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='eval_results_quick.json',
        help='Output JSON file'
    )

    args = parser.parse_args()

    # Set threshold
    prs_threshold = args.prs_threshold if args.prs_threshold is not None else config_settings.prs_threshold

    # Load data
    dataset_dir = Path(args.dataset_dir)
    concepts = load_concepts(dataset_dir)
    ground_truth = load_ground_truth(dataset_dir)

    # Count positive relations
    num_positive = sum(1 for label in ground_truth.values() if label == 1)

    # Initialize embedding client
    logger.info("Initializing embedding client...")
    embedding_client = EmbeddingClient()

    # Generate embeddings
    logger.info("Generating embeddings for concepts...")
    compute_embedding_for_concepts(
        concepts=concepts,
        client=embedding_client
    )

    # Rank prerequisites
    logger.info(f"Ranking prerequisites with threshold={prs_threshold:.3f}...")
    prereq_pairs = rank_prerequisites(
        concepts=concepts,
        prs_threshold=prs_threshold
    )

    logger.info(f"Found {len(prereq_pairs)} candidate pairs")

    # Evaluate
    all_concept_ids = [c.concept_id for c in concepts]
    metrics = evaluate_ranking(prereq_pairs, ground_truth, all_concept_ids)

    # Prepare results
    results = {
        'config': {
            'prs_threshold': prs_threshold,
            'num_concepts': len(concepts),
            'num_total_pairs': len(ground_truth),
            'num_positive_relations': num_positive,
        },
        'metrics': metrics
    }

    # Print and save
    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
