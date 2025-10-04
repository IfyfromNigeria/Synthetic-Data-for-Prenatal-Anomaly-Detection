import os, torch, numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from src.utils import device_auto, ensure_dir
from src.gan.data import WGANDataset
from src.models.wgan_gp.generator import Generator
import umap.umap_ as umap
import matplotlib.pyplot as plt

def extract_features_in_batches(loader, extractor, normalizer, device):
    feats=[]
    with torch.no_grad():
        for x,_ in loader:
            x = F.interpolate(x, size=(299,299), mode='bilinear', align_corners=False)
            x = x.repeat(1,3,1,1)  # grayscale->RGB
            x = normalizer(x)
            f = extractor(x.to(device)).cpu().numpy()
            feats.append(f)
    return np.vstack(feats)

def run_eval(csv_path, checkpoint, out_dir, latent_dim=100, num_classes=2, num_fake=1000, batch_size=32):
    device = device_auto()
    ensure_dir(out_dir)

    real_ds = WGANDataset(csv_path, img_size=64)
    real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=True)

    inc = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    inc.fc = nn.Identity()
    inc.to(device).eval()
    normalizer = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    real_feats = extract_features_in_batches(real_loader, inc, normalizer, device)[:num_fake]

    netG = Generator(latent_dim, num_classes, img_channels=1).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    netG.load_state_dict(ckpt["netG"]); netG.eval()

    fake_imgs=[]; bs=batch_size
    with torch.no_grad():
        for i in range(0, num_fake, bs):
            cur = min(bs, num_fake - i)
            z = torch.randn(cur, latent_dim, device=device)
            y = torch.randint(0, num_classes, (cur,), device=device)
            y1h = F.one_hot(y, num_classes).float()
            f = netG(z, y1h).cpu()
            fake_imgs.append(f)
    fake = torch.cat(fake_imgs, dim=0)

    f_loader = [(fake[i:i+batch_size], None) for i in range(0, fake.size(0), batch_size)]
    fake_feats = extract_features_in_batches(f_loader, inc, normalizer, device)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X = np.vstack([real_feats, fake_feats])
    y = np.array(["Real"]*len(real_feats) + ["Synthetic"]*len(fake_feats))
    emb = reducer.fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(emb[:len(real_feats),0], emb[:len(real_feats),1], s=10, alpha=0.6, label="Real")
    plt.scatter(emb[len(real_feats):,0], emb[len(real_feats):,1], s=10, alpha=0.6, label="Synthetic")
    plt.legend(); plt.title("UMAP of Inception Features (Real vs Synthetic)")
    outp = os.path.join(out_dir,"wgan_gp_umap.png"); plt.tight_layout(); plt.savefig(outp,dpi=300); plt.close()
    print("Saved:", outp)
