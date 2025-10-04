import argparse
from src.train.train_classifier import run_train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--save_dir",  default="models/checkpoints/real_model_artifacts")
    ap.add_argument("--epochs",    type=int, default=40)
    ap.add_argument("--batch",     type=int, default=16)
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    run_train(csv_path={"train":args.train_csv, "val":args.val_csv},
              save_dir=args.save_dir, num_epochs=args.epochs,
              batch_size=args.batch, lr=args.lr, seed=args.seed)
