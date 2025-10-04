import argparse
from src.train.eval_classifier import run_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--save_dir",  default="outputs/eval")
    args = ap.parse_args()
    run_eval(args.test_csv, args.checkpoint, args.save_dir)
