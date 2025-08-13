from typing import Set, Tuple, List, Dict
import networkx as nx

def coverage_precision_recall_cohesion(E_pred: Set[Tuple[int, int]],
                                       E_true: Set[Tuple[int, int]],
                                       V_true: Set[int]):
    if not V_true:
        return 0.0, 0.0, 0.0, 0.0

    # Node coverage
    V_pred = {v for (u, v) in E_pred for v in (u, v)}
    coverage = len(V_pred & V_true) / len(V_true)

    # Precision / Recall (on undirected, deduped edges)
    tp = len(E_pred & E_true)
    precision = tp / len(E_pred) if E_pred else 0.0
    recall = tp / len(E_true) if E_true else 0.0

    # Island Cohesion: weighted recall of GT edges inside each predicted component
    cohesion_vals: List[float] = []
    weight_vals: List[int] = []
    if E_pred and E_true:
        # Build true adjacency once
        adj_true: Dict[int, Set[int]] = {}
        for a, b in E_true:
            adj_true.setdefault(a, set()).add(b)
            adj_true.setdefault(b, set()).add(a)

        G_pred = nx.Graph()
        G_pred.add_edges_from(E_pred)

        for comp in nx.connected_components(G_pred):
            comp = set(comp)
            # Count gt edges internal to this comp by summing adj degrees and halving
            gt_edges = 0
            for u in comp:
                gt_edges += sum(1 for v in adj_true.get(u, ()) if v in comp)
            gt_edges //= 2
            if gt_edges == 0:
                continue

            # Count predicted edges internal to this comp
            pred_intra = 0
            for (u, v) in E_pred:
                if u in comp and v in comp:
                    pred_intra += 1

            cohesion_vals.append(pred_intra / gt_edges)
            weight_vals.append(gt_edges)

    cohesion = (sum(w * c for w, c in zip(weight_vals, cohesion_vals)) / sum(weight_vals)) if weight_vals else 0.0
    return coverage, precision, recall, cohesion