"""How: 上位trajectoryを3日macro optionへ変換し、工程互換性のための署名を作る。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

BLOCK_TURNS = 72
LAST_ACT_STEP = 718
PRODUCTS = ("WHEAT", "MELON", "STRAWBERRY", "TOMATO", "CARROT", "MILK", "WOOL", "EGG", "FERTILIZER")
ANIMALS = ("COW", "SHEEP", "GOOSE")
SEEDS = ("WHEAT", "MELON", "STRAWBERRY", "TOMATO", "CARROT")
UNIT_OPS = ("PASS", "NORTH", "SOUTH", "EAST", "WEST", "PICKUP", "PLACE", "BUILD_PASTURE", "BUILD_COOP", "PLANT", "WATER", "CARE", "FEED", "DROP", "HARVEST", "FERTILIZE", "DIG", "COLLECT_FERTILIZER")
MARKET_OPS = ("NOOP", "SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL", "HIRE", "BUY_LAND")


@dataclass(frozen=True)
class PlanSignature:
    """How: option切替で壊れやすい工程前提を軽量な離散署名として保持する。"""

    block: int
    hand_count_hint: int
    build_count: int
    plant_count: int
    animal_place_count: int
    pickup_count: int
    market_buy_count: int
    market_sell_count: int


def macro_boundary(step: int, block_turns: int = BLOCK_TURNS) -> bool:
    """How: shop更新と位置リセットに揃えた自然な戦略再評価点だけを許可する。"""
    return step >= 0 and step % block_turns == 0 and step <= LAST_ACT_STEP


def _op(command: Any) -> str:
    return str(command[0]) if isinstance(command, list) and command else "EMPTY"


def plan_signature(actions: list[dict[str, Any]], block: int, block_turns: int = BLOCK_TURNS) -> PlanSignature:
    """How: 72手blockの工程依存をaction頻度へ圧縮してswitch maskに使う。"""
    start = block * block_turns
    chunk = actions[start : start + block_turns]
    counts: Counter[str] = Counter()
    hands = []
    for action in chunk:
        hand_actions = action.get("hands") or []
        hands.append(len(hand_actions))
        for cmd in [action.get("farmer") or ["PASS"], *hand_actions]:
            counts[f"unit:{_op(cmd)}"] += 1
        for order in action.get("market") or []:
            counts[f"market:{_op(order)}"] += 1
    return PlanSignature(
        block=block,
        hand_count_hint=int(round(float(np.median(hands)))) if hands else 0,
        build_count=counts["unit:BUILD_PASTURE"] + counts["unit:BUILD_COOP"],
        plant_count=counts["unit:PLANT"],
        animal_place_count=counts["unit:PLACE"],
        pickup_count=counts["unit:PICKUP"],
        market_buy_count=sum(counts[f"market:{x}"] for x in ("BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL", "HIRE", "BUY_LAND")),
        market_sell_count=counts["market:SELL"],
    )


def signature_vector(signature: PlanSignature) -> np.ndarray:
    """How: compatibility auditへ使える小さい固定長ベクトルを返す。"""
    return np.asarray([
        signature.block / 9.0,
        signature.hand_count_hint / 8.0,
        signature.build_count / 72.0,
        signature.plant_count / 72.0,
        signature.animal_place_count / 72.0,
        signature.pickup_count / 72.0,
        signature.market_buy_count / 100.0,
        signature.market_sell_count / 100.0,
    ], dtype=np.float32)


def macro_feature_vector(actions: list[dict[str, Any]], block: int, block_turns: int = BLOCK_TURNS) -> np.ndarray:
    """How: exact tapeではなく工程・商品別売買・資材フローを表すoption embeddingを作る。"""
    start = block * block_turns
    chunk = actions[start : start + block_turns]
    count: Counter[str] = Counter()
    qty: Counter[str] = Counter()
    for action in chunk:
        farmer = action.get("farmer") or ["PASS"]
        count[f"farmer:{_op(farmer)}"] += 1
        hands = action.get("hands") or []
        count["hands:n"] += len(hands)
        for cmd in hands:
            op = _op(cmd)
            count[f"hand:{op}"] += 1
            if isinstance(cmd, list) and len(cmd) > 1:
                count[f"hand_arg:{op}:{cmd[1]}"] += 1
        market = action.get("market") or []
        count["market:n"] += len(market)
        for cmd in market:
            op = _op(cmd)
            count[f"market:{op}"] += 1
            if isinstance(cmd, list) and len(cmd) > 1:
                count[f"market_arg:{op}:{cmd[1]}"] += 1
            if isinstance(cmd, list) and len(cmd) > 2 and isinstance(cmd[2], (int, float)):
                qty[f"qty:{op}:{cmd[1]}"] += float(cmd[2])
    keys: list[str] = []
    for op in UNIT_OPS:
        keys.extend([f"farmer:{op}", f"hand:{op}"])
    keys.extend(["hands:n", "market:n"])
    keys.extend(f"market:{op}" for op in MARKET_OPS)
    for op in ("SELL", "BUY_PRODUCT"):
        for item in PRODUCTS:
            keys.extend([f"market_arg:{op}:{item}", f"qty:{op}:{item}"])
    for item in SEEDS:
        keys.extend([f"market_arg:BUY_SEED:{item}", f"qty:BUY_SEED:{item}"])
    for item in ANIMALS:
        keys.extend([f"market_arg:BUY_ANIMAL:{item}", f"qty:BUY_ANIMAL:{item}"])
    for item in PRODUCTS + ANIMALS:
        for op in ("PICKUP", "PLACE", "PLANT"):
            keys.append(f"hand_arg:{op}:{item}")
    return np.asarray([float(count[k] + qty[k]) for k in keys], dtype=np.float32)


def option_matrix(plans: Iterable[dict[str, Any]], block: int) -> tuple[np.ndarray, list[str]]:
    """How: 任意数expertを同じrich option表現へ変換し、固定class headを避ける。"""
    vectors, ids = [], []
    for plan in plans:
        vectors.append(macro_feature_vector(plan["actions"], block))
        ids.append(str(plan.get("id", len(ids))))
    return np.stack(vectors), ids
