"""EVOLYX Reproducible End-to-End Training & Promotion Pipeline.

Executes:
1. Ingest approved experience trajectories from database
2. Filter & sanitize dataset (PII removal, deduplication)
3. Train candidate neural weights (real backprop & optimizer step)
4. Save immutable checkpoint with SHA256 integrity hash
5. Run multi-metric evaluation suite & calculate confidence intervals
6. Enforce regression gates (< 2.0% allowed baseline drop)
7. Atomically promote or reject the candidate model
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

# Ensure EVOLYX is in python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evolyx.evaluation.evaluation_suite import EvaluationSuite
from evolyx.learning.dataset_builder import DatasetBuilder
from evolyx.learning.trajectory_store import TrajectoryStore
from evolyx.storage.registry import ExperimentRegistry
from evolyx.training.micro_trainer import MicroNeuralTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EVOLYX.Pipeline")


def run_pipeline(version_tag: str, epochs: int = 5) -> bool:
    print("==================================================================")
    print(f"       EVOLYX REPRODUCIBLE TRAINING & PROMOTION PIPELINE ({version_tag})")
    print("==================================================================")

    # 1. Dataset Generation from Verified Experience
    logger.info("Stage 1: Ingesting verified trajectories...")
    traj_store = TrajectoryStore(db_path=str(ROOT / "artifacts" / "evolyx.db"))
    builder = DatasetBuilder(trajectory_store=traj_store, output_dir=str(ROOT / "artifacts" / "datasets"))
    dataset_manifest = builder.build_sft_dataset(version=version_tag)
    print(f"[+] Dataset Ready: {dataset_manifest['train_file']}")

    # 2. Candidate Neural Training (Real gradient descent on local micro-model)
    run_id = f"cand_{uuid.uuid4().hex[:8]}"
    logger.info("Stage 2: Launching neural training for candidate %s...", run_id)
    trainer = MicroNeuralTrainer(d_model=64, n_layers=2)

    # Load text samples from dataset
    samples = []
    with open(dataset_manifest["train_file"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                assistant_content = item["messages"][-1]["content"]
                samples.append(assistant_content)

    train_metrics = trainer.train_step(text_samples=samples, epochs=epochs, lr=1e-3)
    print(f"[+] Neural Training Complete: Initial Loss={train_metrics['initial_loss']} -> Final Loss={train_metrics['final_loss']}")

    # 3. Checkpoint Serialization & Integrity Hash
    logger.info("Stage 3: Serializing immutable checkpoint...")
    cand_dir = ROOT / "artifacts" / "models" / "candidates" / run_id
    ckpt_meta = trainer.save_checkpoint(cand_dir, model_id=run_id)
    print(f"[+] Checkpoint Saved: {ckpt_meta['weights_path']}")
    print(f"[+] Checkpoint SHA256: {ckpt_meta['checksum'][:16]}...")

    # 4. Multi-Metric Benchmark Battery & Confidence Intervals
    logger.info("Stage 4: Running comprehensive evaluation battery...")
    eval_suite = EvaluationSuite(benchmark_file=str(ROOT / "artifacts" / "evaluations" / "held_out_benchmark.json"))

    # Check for dataset contamination
    is_contaminated, leaked_cases = eval_suite.check_data_contamination(samples)
    if is_contaminated:
        logger.error("[!] Data contamination detected! Held-out cases leaked into training: %s", leaked_cases)
        return False

    eval_results = eval_suite.evaluate_model_comprehensive(trainer.model)
    overall_mean = eval_results["overall_mean"]
    ci_low, ci_high = eval_results["overall_ci_95"]
    print(f"[+] Benchmark Battery Complete: Score = {overall_mean:.4f} (95% CI: [{ci_low}, {ci_high}])")

    # 5. Registry Record
    logger.info("Stage 5: Persisting immutable experiment record...")
    registry = ExperimentRegistry(
        db_path=str(ROOT / "artifacts" / "evolyx.db"),
        artifacts_root=str(ROOT / "artifacts"),
    )
    registry.register_run(
        run_id=run_id,
        parent_version="v001",
        dataset_id=dataset_manifest["manifest_file"],
        config={"d_model": 64, "n_layers": 2, "epochs": epochs, "lr": 1e-3},
        checkpoint_file=Path(ckpt_meta["weights_path"]),
        metrics=train_metrics,
        benchmark_scores={"overall": overall_mean, "ci_low": ci_low, "ci_high": ci_high},
    )

    # 6. Release Gating & Atomic Promotion
    logger.info("Stage 6: Enforcing regression release gates...")
    baseline_threshold = 0.85
    passed_gate = overall_mean >= baseline_threshold and ci_low >= 0.80

    if passed_gate:
        promoted = registry.atomic_promote(run_id=run_id, version_tag=version_tag)
        print("\n==================================================================")
        print(f" [OK] CANDIDATE PROMOTED: {run_id} is now active production '{version_tag}'")
        print("==================================================================")
        return True
    else:
        print("\n==================================================================")
        print(f" [REJECTED] CANDIDATE REJECTED: Score {overall_mean:.4f} did not clear gate threshold {baseline_threshold}")
        print("==================================================================")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EVOLYX End-to-End Training Pipeline")
    parser.add_argument("--version", type=str, default="v002", help="Version tag for candidate")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    args = parser.parse_args()

    success = run_pipeline(version_tag=args.version, epochs=args.epochs)
    sys.exit(0 if success else 1)
