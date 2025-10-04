import argparse
from src.gan.eval_wgan_gp import run_eval

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out_dir", default="outputs/gan_eval")
    p.add_argument("--zdim", type=int, default=100)
    p.add_argument("--classes", type=int, default=2)
    p.add_argument("--num_fake", type=int, default=1000)
    args = p.parse_args()
    run_eval(csv_path=args.csv, checkpoint=args.checkpoint, out_dir=args.out_dir,
             latent_dim=args.zdim, num_classes=args.classes, num_fake=args.num_fake)
