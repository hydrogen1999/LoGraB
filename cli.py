import argparse, yaml
from pathlib import Path

from .generation import generate
from .evaluation import eval_reconstruct, eval_nodeclf, eval_linkpred
from .splits import compute_patch_splits


def main():
    p = argparse.ArgumentParser("lograb")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--config", type=Path, required=True)

    e = sub.add_parser("eval")
    e.add_argument("--task", choices=["reconstruct", "nodeclf", "linkpred"], required=True)
    e.add_argument("--instance", type=Path, required=True)
    e.add_argument("--pred", type=Path, required=False)
    # Nodeclf extras (ignored by other tasks)
    e.add_argument("--model", type=str, default="sage")
    e.add_argument("--model-cfg", type=Path, default=None)
    e.add_argument("--epochs", type=int, default=100)
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--lr", type=float, default=1e-2)
    e.add_argument("--wd", type=float, default=5e-4)
    e.add_argument("--checkpoint", type=Path, default=None)

    s = sub.add_parser("splits")
    s.add_argument("--instance", type=Path, required=True)
    s.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.cmd == "generate":
        cfg = yaml.safe_load(args.config.read_text())
        generate(cfg)
    elif args.cmd == "splits":
        compute_patch_splits(args.instance, args.seed)
        print(f"[splits] wrote {args.instance / 'splits.yml'}")
    else:
        if args.task == "reconstruct":
            if args.pred is None:
                p.error("--pred is required for task=reconstruct")
            eval_reconstruct(args.instance, args.pred)
        elif args.task == "linkpred":
            if args.pred is None:
                p.error("--pred is required for task=linkpred")
            eval_linkpred(args.instance, args.pred)
        else:
            # nodeclf
            eval_nodeclf(
                args.instance,
                args.pred or Path("_unused.txt"),
                model=args.model,
                model_cfg=args.model_cfg,
                epochs=args.epochs,
                batch=args.batch,
                lr=args.lr,
                wd=args.wd,
                checkpoint=args.checkpoint,
            )


if __name__ == "__main__":
    main()