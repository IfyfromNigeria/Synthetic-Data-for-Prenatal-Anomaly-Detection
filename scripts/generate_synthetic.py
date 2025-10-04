import argparse
from src.gan.generate_dataset import generate_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_normal", type=int, default=555)
    p.add_argument("--num_anomalous", type=int, default=555)
    p.add_argument("--zdim", type=int, default=100)
    p.add_argument("--classes", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    generate_dataset(args.checkpoint, args.out_dir, args.num_normal, args.num_anomalous,
                     latent_dim=args.zdim, num_classes=args.classes, batch_size=args.batch)
