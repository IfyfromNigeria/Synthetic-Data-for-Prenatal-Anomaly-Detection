import argparse, yaml
from src.gan.generate_dataset import generate_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        C = yaml.safe_load(f)

    G = C.get("generator", {})
    paths = C.get("paths", {})
    generate_dataset(
        checkpoint=G.get("checkpoint", "models/checkpoints/wgan_gp_checkpoint.pt"),
        out_dir=G.get("output_dir", "outputs/samples"),
        num_normal=G.get("num_normal", 500),
        num_anomalous=G.get("num_anomalous", 500),
        latent_dim=G.get("z_dim", 100),
        num_classes=G.get("num_classes", 2),
        batch_size=G.get("batch_size", 32),
    )
