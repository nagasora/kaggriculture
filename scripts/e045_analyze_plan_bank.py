"""How: 最新上位replay bankの3日macro一貫性を定量化する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from kaggriculture_e045.macro import BLOCK_TURNS, option_matrix


def exact_similarity(a: list[dict], b: list[dict]) -> float:
    n = min(len(a), len(b))
    return 0.0 if n == 0 else sum(x == y for x, y in zip(a[:n], b[:n])) / n


def lcp(a: list[dict], b: list[dict]) -> int:
    out = 0
    for x, y in zip(a, b):
        if x != y:
            break
        out += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads(args.bank.read_text())
    same_lcp, cross_lcp = [], []
    block_rows = []
    for block in range(10):
        same, cross = [], []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                is_same = rows[i]["family"] == rows[j]["family"]
                if block == 0:
                    (same_lcp if is_same else cross_lcp).append(lcp(rows[i]["actions"], rows[j]["actions"]))
                s = block * BLOCK_TURNS
                sim = exact_similarity(rows[i]["actions"][s : s + BLOCK_TURNS], rows[j]["actions"][s : s + BLOCK_TURNS])
                (same if is_same else cross).append(sim)
        X, _ = option_matrix(rows, block)
        y = np.asarray([r["family"] for r in rows])
        pred = []
        for i in range(len(rows)):
            train = np.arange(len(rows)) != i
            scaler = StandardScaler().fit(X[train])
            Xt = scaler.transform(X[train])
            xi = scaler.transform(X[i : i + 1])[0]
            centers = {f: Xt[y[train] == f].mean(0) for f in sorted(set(y[train]))}
            pred.append(min(centers, key=lambda f: float(np.linalg.norm(xi - centers[f]))))
        block_rows.append({
            "block": block,
            "same_exact_mean": float(np.mean(same)),
            "cross_exact_mean": float(np.mean(cross)),
            "signature_family_loo_accuracy": float(accuracy_score(y, pred)),
        })
    result = {
        "n": len(rows),
        "block_turns": BLOCK_TURNS,
        "same_family_lcp_mean": float(np.mean(same_lcp)),
        "same_family_lcp_median": float(np.median(same_lcp)),
        "cross_family_lcp_mean": float(np.mean(cross_lcp)),
        "cross_family_lcp_median": float(np.median(cross_lcp)),
        "blocks": block_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
