import argparse, yaml
from pathlib import Path

from .generation import generate
from .evaluation import eval_reconstruct, eval_nodeclf, eval_linkpred


def main():
    p = argparse.ArgumentParser("lograb")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--config", type=Path, required=True)

    e = sub.add_parser("eval")
    e.add_argument("--task", choices=["reconstruct", "nodeclf", "linkpred"], required=True)
    e.add_argument("--instance", type=Path, required=True)
    e.add_argument("--pred", type=Path, required=True)

    args = p.parse_args()
    if args.cmd == "generate":
        cfg = yaml.safe_load(args.config.read_text())
        generate(cfg)
    else:
        {
            "reconstruct": eval_reconstruct,
            "nodeclf": eval_nodeclf,
            "linkpred": eval_linkpred,
        }[args.task](args.instance, args.pred)


if __name__ == "__main__":
    main()