"""
Sweep through different PRS thresholds to find optimal value.

This script runs evaluation with multiple thresholds and compares results.
"""
import pandas as pd
from loguru import logger
from embed.prereq_ranking import rank_prerequisites
from embed.embeddings import compute_embedding_for_concepts
from embed.embedding_client import EmbeddingClient
from eval_lecture_bank_quick import (
    load_concepts,
    load_ground_truth,
    evaluate_ranking
)
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def sweep_thresholds(
    concepts,
    ground_truth,
    all_concept_ids,
    thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
):
    """Run evaluation with different thresholds."""
    results = []

    for threshold in thresholds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing threshold: {threshold:.2f}")
        logger.info(f"{'='*60}")

        # Rank prerequisites
        prereq_pairs = rank_prerequisites(
            concepts=concepts,
            prs_threshold=threshold
        )

        # Evaluate
        metrics = evaluate_ranking(prereq_pairs, ground_truth, all_concept_ids)

        # Store results
        result = {
            'threshold': threshold,
            'num_predicted': metrics['num_predicted'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
            'true_positives': metrics['true_positives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
        }
        results.append(result)

        # Print summary
        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Predicted: {metrics['num_predicted']}")
        print(
            f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(
            f"  Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1: {metrics['f1']:.4f}")
        print(
            f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")

    return results


def main():
    """Main function."""
    dataset_dir = Path("dataset/lecture_bank")

    # Load data
    logger.info("Loading dataset...")
    concepts = load_concepts(dataset_dir)
    ground_truth = load_ground_truth(dataset_dir)
    all_concept_ids = [c.concept_id for c in concepts]

    # Initialize embedding client
    logger.info("Initializing SciBERT embedding client...")
    embedding_client = EmbeddingClient(
        model_name="allenai/scibert_scivocab_uncased",
        use_vi_tokenizer=False
    )

    # Generate embeddings
    logger.info("Generating embeddings...")
    compute_embedding_for_concepts(
        concepts=concepts,
        client=embedding_client
    )

    # Sweep thresholds
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    results = sweep_thresholds(
        concepts, ground_truth, all_concept_ids, thresholds)

    # Save results
    output_path = "threshold_sweep_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")

    # Create summary table
    df = pd.DataFrame(results)
    df = df.round(4)

    print("\n" + "="*80)
    print("THRESHOLD SWEEP SUMMARY")
    print("="*80)
    print(df.to_string(index=False))

    # Find best thresholds
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS")
    print("="*80)

    best_f1_idx = df['f1'].idxmax()
    print(
        f"\n🏆 Best F1 Score: {df.loc[best_f1_idx, 'f1']:.4f} at threshold {df.loc[best_f1_idx, 'threshold']:.2f}")
    print(
        f"   Precision: {df.loc[best_f1_idx, 'precision']:.4f}, Recall: {df.loc[best_f1_idx, 'recall']:.4f}")

    best_precision_idx = df['precision'].idxmax()
    print(
        f"\n🎯 Best Precision: {df.loc[best_precision_idx, 'precision']:.4f} at threshold {df.loc[best_precision_idx, 'threshold']:.2f}")
    print(
        f"   Recall: {df.loc[best_precision_idx, 'recall']:.4f}, F1: {df.loc[best_precision_idx, 'f1']:.4f}")

    best_recall_idx = df['recall'].idxmax()
    print(
        f"\n🔍 Best Recall: {df.loc[best_recall_idx, 'recall']:.4f} at threshold {df.loc[best_recall_idx, 'threshold']:.2f}")
    print(
        f"   Precision: {df.loc[best_recall_idx, 'precision']:.4f}, F1: {df.loc[best_recall_idx, 'f1']:.4f}")

    # Save CSV for easy analysis
    csv_path = "threshold_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV results saved to {csv_path}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
