import os, torch
import torch.nn.functional as F
from torchvision.utils import save_image
from src.utils import ensure_dir, device_auto
from src.models.wgan_gp.generator import Generator

def generate_dataset(checkpoint, out_dir, num_normal, num_anomalous, latent_dim=100, num_classes=2, batch_size=32):
    device = device_auto()
    ensure_dir(os.path.join(out_dir, "normal"))
    ensure_dir(os.path.join(out_dir, "anomalous"))

    netG = Generator(latent_dim, num_classes, img_channels=1).to(device)
    state = torch.load(checkpoint, map_location=device)
    netG.load_state_dict(state["netG"]); netG.eval()

    plan = {0: ("normal", num_normal), 1: ("anomalous", num_anomalous)}
    with torch.no_grad():
        for cls,(name, count) in plan.items():
            done=0
            while done < count:
                cur = min(batch_size, count-done)
                z = torch.randn(cur, latent_dim, device=device)
                y = torch.full((cur,), cls, dtype=torch.long, device=device)
                y1h = F.one_hot(y, num_classes).float()
                imgs = netG(z, y1h).cpu()
                for i in range(cur):
                    save_image(imgs[i], os.path.join(out_dir, name, f"synthetic_{name}_{done+i:04d}.png"), normalize=True)
                done += cur
            print(f"Generated {count} images for {name}.")
