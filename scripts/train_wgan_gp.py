import argparse
from src.gan.train_wgan_gp import run_train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", default="models/wgan_gp")
    p.add_argument("--epochs", type=int, default=1501)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--zdim", type=int, default=100)
    p.add_argument("--classes", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--critic_updates", type=int, default=5)
    p.add_argument("--img_interval", type=int, default=50)
    args = p.parse_args()

    run_train(csv_path=args.csv, out_dir=args.out_dir, epochs=args.epochs,
              batch_size=args.batch, latent_dim=args.zdim, num_classes=args.classes,
              lr=args.lr, critic_updates=args.critic_updates, img_interval=args.img_interval)
