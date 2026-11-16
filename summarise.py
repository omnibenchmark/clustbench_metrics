#!/usr/bin/env python
"""Gather entrypoint (api 0.8.0): one tidy table of the scores of a group.

Reads the members from --from-files (lineage.json, version 2) and labels every
row from the member's lineage, never from its path.
"""

import argparse
import csv
import gzip
import json

COLUMNS = ["dataset_generator", "dataset", "method", "method_params", "metric",
           "k_position", "k", "score"]
# the clustering modules always emit true k - 2 .. true k + 2, in this order
K_POSITIONS = ["k-2", "k-1", "k", "k+1", "k+2"]


def rows(member):
    by_stage = {entry["stage"]: entry for entry in member["lineage"]}
    data, method = by_stage["data"], by_stage["clustering"]
    label = {
        "dataset_generator": data["parameters"]["dataset_generator"],
        "dataset": data["parameters"]["dataset_name"],
        "method": method["module"],
        "method_params": ";".join(f"{k}={v}" for k, v in sorted(method["parameters"].items())),
        "metric": by_stage["metrics"]["parameters"]["metric"],
    }
    with gzip.open(member["files"]["metrics_scores"], "rt") as fh:
        ks, scores = (line.strip().split(",") for line in fh.read().splitlines()[:2])
    if len(ks) != len(K_POSITIONS):
        raise ValueError(f"{member['node_id']}: expected {len(K_POSITIONS)} ks, got {ks}")
    for position, k, score in zip(K_POSITIONS, ks, scores):
        yield {**label, "k_position": position, "k": k.removeprefix("k="), "score": score}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-files", required=True, help="gather manifest (lineage.json)")
    parser.add_argument("--output", action="append", default=[], help="id=PATH")
    args, _ = parser.parse_known_args()
    outputs = dict(o.split("=", 1) for o in args.output)

    manifest = json.load(open(args.from_files))
    if manifest.get("version") != 2:
        raise ValueError(f"unsupported manifest version: {manifest.get('version')}")
    table = sorted(
        (row for member in manifest["members"] for row in rows(member)),
        key=lambda r: [r[c] for c in COLUMNS[:5]] + [K_POSITIONS.index(r["k_position"])],
    )
    with open(outputs["metrics_summary"], "w", newline="") as fh:
        writer = csv.DictWriter(fh, COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
