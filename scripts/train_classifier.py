import argparse, yaml
from src.train.train_classifier import run_train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        C = yaml.safe_load(f)

    run_train(
        csv_path={"train": C["paths"]["train_csv"], "val": C["paths"]["val_csv"]},
        save_dir=C["paths"].get("checkpoints", "models/checkpoints"),
        num_epochs=C["training"].get("epochs", 40),
        batch_size=C["training"].get("batch_size", 16),
        lr=C["training"].get("learning_rate", 1e-4),
        seed=C.get("seed", 42),
    )
