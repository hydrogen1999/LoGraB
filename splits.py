from __future__ import annotations
import yaml, gzip, json, random
from pathlib import Path
from typing import Dict, List, Set


def _load_patch_ids(instance: Path) -> List[str]:
    ids = []
    with gzip.open(instance / "patches.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
    return ids


def _load_patch_nodes(instance: Path) -> Dict[str, Set[int]]:
    mp: Dict[str, Set[int]] = {}
    with gzip.open(instance / "patches.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            mp[obj["id"]] = set(int(x) for x in obj["nodes_global"])
    return mp


def compute_patch_splits(instance: Path, seed: int = 42, ratios=(0.6, 0.2, 0.2)):
    """Assign patches to train/val/test, keeping overlapping patches in the *same* split.
    Greedy packing over the patch-overlap graph for disjointness priority.
    """
    rnd = random.Random(seed)
    ids = _load_patch_ids(instance)
    nodes = _load_patch_nodes(instance)

    # Build overlap graph (patch adjacency if share any node)
    neigh = {pid: set() for pid in ids}
    by_node: Dict[int, List[str]] = {}
    for pid, S in nodes.items():
        for u in S:
            by_node.setdefault(u, []).append(pid)
    for u, lst in by_node.items():
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                a, b = lst[i], lst[j]
                neigh[a].add(b); neigh[b].add(a)

    # Connected components in patch graph
    unvis = set(ids)
    comps: List[List[str]] = []
    while unvis:
        start = unvis.pop()
        q = [start]
        comp = [start]
        while q:
            x = q.pop()
            for y in neigh[x]:
                if y in unvis:
                    unvis.remove(y)
                    q.append(y)
                    comp.append(y)
        comps.append(comp)

    rnd.shuffle(comps)
    n = len(ids)
    target = [int(n*ratios[0]), int(n*ratios[1])]
    train, val, test = [], [], []
    for comp in comps:
        if len(train) < target[0]:
            train += comp
        elif len(val) < target[1]:
            val += comp
        else:
            test += comp

    # Persist
    out = {
        "seed": seed,
        "ratios": list(ratios),
        "train_patches": sorted(train),
        "val_patches": sorted(val),
        "test_patches": sorted(test),
    }
    (instance / "splits.yml").write_text(yaml.safe_dump(out, sort_keys=False))
    return out
