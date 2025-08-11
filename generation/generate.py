from pathlib import Path
from typing import Dict, Any, List
import json, yaml, numpy as np, torch
from tqdm import tqdm
from torch_geometric.datasets import Planetoid

from ..utils import set_global_seed, write_jsonl_gz, sha256
from .d_hop import get_d_hop_patch
from .cluster import partition_metis, build_cluster_patches
from .random import random_seed_patches
from .spectral import spectral_embed


def generate(cfg: Dict[str, Any]):
    """Create one LoGraB instance from *cfg* dict."""
    ds_name = cfg["dataset_name"]
    strategy = cfg["strategy"]
    d, k, sigma, p, seed = cfg["d"], cfg["k"], cfg["sigma"], cfg["p"], cfg["seed"]

    set_global_seed(seed)

    tag = f"{strategy}_d{d}_k{k}_s{sigma:.2f}_p{p}"
    root = Path(cfg["root_dir"]) / ds_name / tag
    patches_dir = root / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    # load graph via PyG
    data_dir = Path("source_data") / ds_name
    dataset = Planetoid(str(data_dir), name=ds_name)
    data = dataset[0]
    num_nodes, edge_index = data.num_nodes, data.edge_index

    # choose observed vertices (coverage p)
    observed_mask = np.random.rand(num_nodes) < p

    # ---------------- generate patches ------------------------------------
    all_rows: List[Dict[str, Any]] = []

    if strategy == "d-hop":
        nodes_iter = np.nonzero(observed_mask)[0]
        for v in tqdm(nodes_iter, desc="d-hop patches"):
            patch = get_d_hop_patch(int(v), d, edge_index, num_nodes)
            spec = spectral_embed(patch["edge_index"], len(patch["nodes"],), k, sigma)
            if spec is None:
                continue
            row = {
                "id": f"patch_{int(v)}",
                "nodes_global": patch["nodes"].tolist(),
                "eigvec": spec["eigvec"].tolist(),
                "eigval": spec["eigval"].tolist(),
                "center_node_global": int(v),
                "strategy": strategy,
                "local_to_global_map": patch["local2global"],
            }
            all_rows.append(row)
    elif strategy == "cluster":
        parts = max(32, num_nodes // 512)
        mapping = partition_metis(data, parts)
        patches = build_cluster_patches(data, mapping)
        for idx, patch in enumerate(tqdm(patches, desc="cluster patches")):
            if np.random.rand() > p:
                continue
            spec = spectral_embed(edge_index[:, torch.isin(edge_index[0], patch["nodes"])]
                                  , len(patch["nodes"]), k, sigma)
            if spec is None:
                continue
            row = {
                "id": f"patch_{idx}",
                "nodes_global": patch["nodes"].tolist(),
                "eigvec": spec["eigvec"].tolist(),
                "eigval": spec["eigval"].tolist(),
                "strategy": strategy,
                "local_to_global_map": {str(i): int(n) for i, n in enumerate(patch["nodes"])},
            }
            all_rows.append(row)
    elif strategy == "random":
        seeds = max(int(0.2 * num_nodes), 256)
        for patch in tqdm(random_seed_patches(num_nodes, seeds, d, edge_index), desc="random patches"):
            if np.random.rand() > p:
                continue
            spec = spectral_embed(patch["edge_index"], len(patch["nodes"]), k, sigma)
            if spec is None:
                continue
            row = {
                "id": f"patch_{len(all_rows)}",
                "nodes_global": patch["nodes"].tolist(),
                "eigvec": spec["eigvec"].tolist(),
                "eigval": spec["eigval"].tolist(),
                "center_node_global": patch.get("center"),
                "strategy": strategy,
                "local_to_global_map": patch["local2global"],
            }
            all_rows.append(row)
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    # ------ write files ----------------------------------------------------
    write_jsonl_gz(root / "patches.jsonl.gz", all_rows)

    metadata = {
        "dataset": ds_name,
        "strategy": strategy,
        "parameters": {"d": d, "k": k, "sigma": sigma, "p": p, "seed": seed},
        "num_patches": len(all_rows),
        "avg_patch_size": float(np.mean([len(r["nodes_global"]) for r in all_rows])),
        "patches_checksum": sha256(root / "patches.jsonl.gz"),
    }
    (root / "metadata.yml").write_text(yaml.safe_dump(metadata, sort_keys=False))

    (root / "README.md").write_text(
        f"# LoGraB instance\n\n- Dataset: {ds_name}\n- Tag: {tag}\n- Params: d={d}, k={k}, σ={sigma}, p={p}\n- Patches: {len(all_rows)}\n")

