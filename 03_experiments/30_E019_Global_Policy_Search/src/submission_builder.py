"""Build standalone Kaggle submission archives for E019 promoted candidates.

How:
- Bundle the exact candidate spec, inference weights, E015 executor code, and E018 residual assets.
- Reuse the E015 runtime-safe bundle-root resolver so Kaggle exec loading does not require __file__.

Why not:
- Do not package training code or discover Drive files at inference time.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tarfile

from .candidate_space import CandidateSpec

MAIN_TEMPLATE = r'''"""How: load one frozen E019 promoted candidate from its own submission bundle."""
from __future__ import annotations
import json
import sys
from pathlib import Path
_AGENT = None

def _bundle_root(configuration):
    """How: resolve colocated assets even when Kaggle exec does not define __file__."""
    raw_path = configuration.get("__raw_path__") if configuration is not None and hasattr(configuration, "get") else None
    sources = (globals().get("__file__"), _bundle_root.__code__.co_filename, raw_path)
    for source in sources:
        if not isinstance(source, str) or not source or source.startswith("<") or "\n" in source:
            continue
        p = Path(source)
        if p.is_file():
            root = p.resolve().parent
            if (root / "candidate.json").is_file() and (root / "e019" / "agents.py").is_file():
                return root
    raise FileNotFoundError("E019 bundle root could not be resolved")

def _load(configuration):
    root = _bundle_root(configuration)
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(root))
        from e019.candidate_space import CandidateSpec
        from e019.agents import AssetContext, build_candidate
        raw = json.loads((root / "candidate.json").read_text(encoding="utf-8"))
        raw["route"] = tuple(raw.get("route") or ())
        raw["phase_days"] = tuple(raw.get("phase_days") or (0, 6, 12, 18, 24))
        spec = CandidateSpec(**raw)
        payload = json.loads((root / "route_policy.json").read_text(encoding="utf-8")) if (root / "route_policy.json").is_file() else None
        ctx = AssetContext(root, root / "e018d1", root / "replay_families")
        return build_candidate(spec, ctx, seed=0, route_policy_payload=payload)
    finally:
        sys.path[:] = previous

def agent(observation, configuration):
    global _AGENT
    if _AGENT is None:
        _AGENT = _load(configuration)
    return _AGENT(observation, configuration)
'''


def build_submission(*, spec: CandidateSpec, e015_root: Path, e018_agent_dir: Path, replay_model_dir: Path,
                     ppo_dir: Path, source_dir: Path, out_dir: Path) -> Path:
    stage = out_dir / f"{spec.candidate_id}_bundle"
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for package in ("e014b", "e014b2", "e014b3", "e015"):
        shutil.copytree(e015_root / package, stage / package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(e015_root / "inputs", stage / "inputs", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (stage / "runs" / "pilot").mkdir(parents=True, exist_ok=True)
    shutil.copy2(e015_root / "runs" / "pilot" / "policy.json", stage / "runs" / "pilot" / "policy.json")
    shutil.copytree(source_dir, stage / "e019", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (stage / "e019" / "__init__.py").touch(exist_ok=True)
    shutil.copytree(e018_agent_dir, stage / "e018d1", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    replay_dst = stage / "replay_families"; replay_dst.mkdir()
    shutil.copy2(replay_model_dir / "day_family_profiles.csv", replay_dst / "day_family_profiles.csv")
    (stage / "candidate.json").write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    ppo = ppo_dir / f"{spec.candidate_id}.json"
    if ppo.exists(): shutil.copy2(ppo, stage / "route_policy.json")
    (stage / "main.py").write_text(MAIN_TEMPLATE, encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / f"{spec.candidate_id}_submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for p in sorted(stage.rglob("*")):
            if p.is_file(): tf.add(p, arcname=str(p.relative_to(stage)))
    return archive
