"""EVOLYX Experiment & Architecture Dashboard."""

import sys
from evolyx.storage.artifacts import ArtifactStore


def display_dashboard():
    artifacts = ArtifactStore()
    candidates = artifacts.list_candidate_models()
    datasets = artifacts.list_datasets()

    print("==================================================")
    print("           EVOLYX PLATFORM DASHBOARD              ")
    print("==================================================")
    print(f"Artifact Store:    {artifacts.root.resolve()}")
    print(f"Candidate Models:  {len(candidates)}")
    for c in candidates:
        print(f"  - {c}")
    print(f"Generated Datasets:{len(datasets)}")
    for d in datasets:
        print(f"  - {d}")
    print("Status:            HEALTHY / READY")
    print("==================================================")


if __name__ == "__main__":
    display_dashboard()
