import argparse, yaml
from src.train.eval_classifier import run_eval

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        C = yaml.safe_load(f)

    run_eval(
        test_csv=C["paths"]["test_csv"],
        checkpoint_path=C["model"]["checkpoint"],
        save_dir=C["paths"].get("eval_out", "outputs/eval"),
    )
