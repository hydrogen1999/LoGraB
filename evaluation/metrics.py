from typing import Set, Tuple, List
import networkx as nx

def coverage_precision_recall_cohesion(E_pred: Set[Tuple[int, int]],
                                       E_true: Set[Tuple[int, int]],
                                       V_true: Set[int]):
    if not V_true:
        return 0.0, 0.0, 0.0, 0.0

    V_pred = {v for (u, v) in E_pred for v in (u, v)}
    coverage = len(V_pred & V_true) / len(V_true)

    tp = len(E_pred & E_true)
    precision = tp / len(E_pred) if E_pred else 0.0
    recall = tp / len(E_true) if E_true else 0.0

    cohesion_vals: List[float] = []
    weight_vals: List[int] = []
    if E_pred and E_true:
        G_pred = nx.Graph()
        G_pred.add_edges_from(E_pred)
        for comp in nx.connected_components(G_pred):
            comp = set(comp)
            gt_edges = {(min(u, v), max(u, v)) for u in comp for v in comp if u < v and ((u, v) in E_true)}
            if not gt_edges:
                continue
            pred_edges = {(min(u, v), max(u, v)) for (u, v) in E_pred if u in comp and v in comp}
            cohesion_vals.append(len(pred_edges & gt_edges) / len(gt_edges))
            weight_vals.append(len(gt_edges))
    cohesion = (sum(w * c for w, c in zip(weight_vals, cohesion_vals)) / sum(weight_vals)) if weight_vals else 0.0

    return coverage, precision, recall, cohesion