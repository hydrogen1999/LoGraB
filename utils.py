import gzip, hashlib, json, random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def sha256(path: Path, chunk: int = 4096) -> str:
    """Return SHA‑256 hex digest for *path*."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def write_jsonl_gz(path: Path, rows: List[Dict[str, Any]]):
    """Write *rows* list to *.jsonl.gz* file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf‑8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)