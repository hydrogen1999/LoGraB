from pathlib import Path
from typing import Dict, Any, List
import yaml, numpy as np, torch
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph

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
    laplacian = cfg.get("laplacian", "unnormalized")

    set_global_seed(seed)

    tag = f"{strategy}_d{d}_k{k}_s{sigma:.2f}_p{p}_{laplacian[0]}"
    root = Path(cfg["root_dir"]) / ds_name / tag
    root.mkdir(parents=True, exist_ok=True)

    # load graph via PyG
    data_dir = Path("source_data") / ds_name
    dataset = Planetoid(str(data_dir), name=ds_name)
    data = dataset[0]
    num_nodes, edge_index = data.num_nodes, data.edge_index

    # choose observed vertices (coverage p) – node-level
    observed_mask = np.random.rand(num_nodes) < float(p)

    all_rows: List[Dict[str, Any]] = []

    if strategy == "d-hop":
        nodes_iter = np.nonzero(observed_mask)[0]
        for v in tqdm(nodes_iter, desc="d-hop patches"):
            patch = get_d_hop_patch(int(v), d, edge_index, num_nodes)
            n_local = int(patch["nodes"].numel())
            spec = spectral_embed(patch["edge_index"], n_local, k, sigma, laplacian=laplacian)
            if spec is None:
                continue
            row = {
                "id": f"patch_{int(v)}",
                "nodes_global": patch["nodes"].tolist(),
                "eigvec": spec["eigvec"].tolist(),
                "eigval": spec["eigval"].tolist(),
                "center_node_global": int(v),
                "strategy": strategy,
                "local_to_global_map": {str(l): int(g) for l, g in patch["local2global"].items()},
            }
            all_rows.append(row)

    elif strategy == "cluster":
        # cluster-level coverage (skip some patches) while still respecting node-level realism
        parts = max(32, num_nodes // 512)
        mapping = partition_metis(data, parts)
        patches = build_cluster_patches(data, mapping)
        for idx, patch in enumerate(tqdm(patches, desc="cluster patches")):
            if np.random.rand() > float(p):
                continue
            # extract local subgraph and relabel
            sub_ei, _, node_map = subgraph(patch["nodes"], edge_index, relabel_nodes=True)
            n_local = int(patch["nodes"].numel())
            spec = spectral_embed(sub_ei, n_local, k, sigma, laplacian=laplacian)
            if spec is None:
                continue
            l2g = {str(int(l)): int(patch["nodes"][int(l)]) for l in range(n_local)}
            row = {
                "id": f"patch_{idx}",
                "nodes_global": patch["nodes"].tolist(),
                "eigvec": spec["eigvec"].tolist(),
                "eigval": spec["eigval"].tolist(),
                "strategy": strategy,
                "local_to_global_map": l2g,
            }
            all_rows.append(row)

    elif strategy == "random":
        seeds = max(int(0.2 * num_nodes), 256)
        for patch in tqdm(random_seed_patches(num_nodes, seeds, d, edge_index), desc="random patches"):
            if np.random.rand() > float(p):
                continue
            n_local = int(patch["nodes"].numel())
            spec = spectral_embed(patch["edge_index"], n_local, k, sigma, laplacian=laplacian)
            if spec is None:
                continue
            row = {
                "id": f"patch_{len(all_rows)}",
                "nodes_global": patch["nodes"].tolist(),
                "eigvec": spec["eigvec"].tolist(),
                "eigval": spec["eigval"].tolist(),
                "center_node_global": patch.get("center"),
                "strategy": strategy,
                "local_to_global_map": {str(int(l)): int(g) for l, g in patch["local2global"].items()},
            }
            all_rows.append(row)
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    # ------ write files ----------------------------------------------------
    patches_path = root / "patches.jsonl.gz"
    write_jsonl_gz(patches_path, all_rows)

    import numpy as np
    metadata = {
        "dataset": ds_name,
        "strategy": strategy,
        "parameters": {"d": d, "k": k, "sigma": sigma, "p": p, "seed": seed, "laplacian": laplacian},
        "num_patches": len(all_rows),
        "avg_patch_size": float(np.mean([len(r["nodes_global"]) for r in all_rows])) if all_rows else 0.0,
        "patches_checksum": sha256(patches_path),
        "source_data_dir": str(data_dir),
        "num_nodes": int(num_nodes),
        "num_edges": int(edge_index.size(1)),
    }
    (root / "metadata.yml").write_text(yaml.safe_dump(metadata, sort_keys=False))

    (root / "README.md").write_text(
        f"# LoGraB instance\n\n- Dataset: {ds_name}\n- Tag: {tag}\n"
        f"- Params: d={d}, k={k}, \u03C3={sigma}, p={p}, L={laplacian}\n- Patches: {len(all_rows)}\n"
    )