#!/usr/bin/env python
"""Gather entrypoint (api 0.8.0): every metric over every clustering of one dataset.

Reads the clusterings from --from-files (lineage.json, version 2), scores each
against --data_true_labels with every metric in --metrics, and writes one tidy
table. Rows are labelled from the member's lineage, never from its path.

A clustering has one column per k, true k - 2 .. true k + 2. Noise points
(true label 0) are not scored, as in Marek Gagolewski's clustbench/score.py:
https://github.com/gagolews/clustering-benchmarks/blob/0e751cc9dfc30f332ea3e3aac2b95ada8fbc266a/clustbench/score.py#L30
"""

import argparse
import csv
import json

import numpy as np
from genieclust import compare_partitions

## partition-comparison (predicted vs true); 'normalized_confusion_matrix' is not a score
METRICS = {name: getattr(compare_partitions, name) for name in [
    "adjusted_fm_score",
    "adjusted_mi_score",
    "adjusted_rand_score",
    "fm_score",
    "mi_score",
    "normalized_clustering_accuracy",
    "normalized_mi_score",
    "normalized_pivoted_accuracy",
    "pair_sets_index",
    "rand_score",
]}

COLUMNS = ["dataset_generator", "dataset", "method", "method_params", "metric",
           "k_position", "k", "score"]
# the clustering modules always emit true k - 2 .. true k + 2, in this order
K_POSITIONS = ["k-2", "k-1", "k", "k+1", "k+2"]


def rows(member, truth, metrics):
    by_stage = {entry["stage"]: entry for entry in member["lineage"]}
    data, method = by_stage["data"], by_stage["clustering"]
    label = {
        "dataset_generator": data["parameters"]["generator"],
        "dataset": data["parameters"]["dataset"],
        "method": method["module"],
        "method_params": ";".join(f"{k}={v}" for k, v in sorted(method["parameters"].items())),
    }
    table = np.loadtxt(member["files"]["clustering_predicted_ks_range"],
                       ndmin=2, dtype=str, delimiter=",")
    ks, predicted = table[0], table[1:].astype(int)
    if len(ks) != len(K_POSITIONS):
        raise ValueError(f"{member['node_id']}: expected {len(K_POSITIONS)} ks, got {ks}")
    true_k = int(truth.max())
    if predicted.min() < 1 or predicted.max() > true_k + 2:
        raise ValueError(f"{member['node_id']}: labels not between 1 and k={true_k} + 2, "
                         f"min is {predicted.min()} and max is {predicted.max()}.")
    scored = truth > 0
    for name in metrics:
        # the fcps module quotes its header ("k=2")
        for position, k, column in zip(K_POSITIONS, ks, predicted.T):
            yield {**label, "metric": name, "k_position": position,
                   "k": k.strip('"').removeprefix("k="),
                   "score": float(METRICS[name](truth[scored], column[scored]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-files", required=True, help="gather manifest (lineage.json)")
    parser.add_argument("--data_true_labels", required=True,
                        help="gz-compressed textfile containing the true labels.")
    parser.add_argument("--metrics", required=True,
                        help="comma-separated metric names: " + ", ".join(METRICS))
    parser.add_argument("--output", action="append", default=[], help="id=PATH")
    args, _ = parser.parse_known_args()
    outputs = dict(o.split("=", 1) for o in args.output)

    metrics = args.metrics.split(",")
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; valid: {sorted(METRICS)}")
    truth = np.loadtxt(args.data_true_labels, ndmin=1)
    if truth.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")
    manifest = json.load(open(args.from_files))
    if manifest.get("version") != 2:
        raise ValueError(f"unsupported manifest version: {manifest.get('version')}")

    table = sorted(
        (row for member in manifest["members"] for row in rows(member, truth, metrics)),
        key=lambda r: [r[c] for c in COLUMNS[:5]] + [K_POSITIONS.index(r["k_position"])],
    )
    with open(outputs["metrics_summary"], "w", newline="") as fh:
        writer = csv.DictWriter(fh, COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
