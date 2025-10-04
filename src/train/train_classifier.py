import os, torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from src.utils import set_seed, ensure_dir, save_history_json, device_auto
from src.data.datasets import ImageCSVClassificationDataset
from src.data.transforms import get_train_transform_en, get_val_transform_en
from src.models.efficientnet_heads import EfficientNetBinaryHead, unfreeze_schedule

def run_train(csv_path, save_dir, num_epochs=40, batch_size=16, lr=1e-4, seed=42):
    set_seed(seed)
    device = device_auto()
    ensure_dir(save_dir)

    train_csv = csv_path["train"]
    val_csv   = csv_path["val"]

    train_ds = ImageCSVClassificationDataset(train_csv, transform=get_train_transform_en())
    val_ds   = ImageCSVClassificationDataset(val_csv,   transform=get_val_transform_en())

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = EfficientNetBinaryHead().to(device)
    opt = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    history = {k:[] for k in ["train_loss","val_loss","train_acc","val_acc","train_f1","val_f1","train_auc","val_auc"]}

    for epoch in range(1, num_epochs+1):
        unfreeze_schedule(model, epoch)

        # ---- train
        model.train()
        tloss=0.0; tpred=[]; tprob=[]; tlabels=[]
        for x,y in tqdm(train_loader, desc=f"Train {epoch}/{num_epochs}"):
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            tloss += loss.item()*x.size(0)
            p = torch.sigmoid(out).detach().cpu().numpy().flatten()
            tprob.extend(p); tpred.extend((p>0.5).astype(int)); tlabels.extend(y.cpu().numpy().flatten())
    # metrics
        from sklearn.metrics import roc_auc_score
        tacc = accuracy_score(tlabels, tpred); tf1 = f1_score(tlabels, tpred)
        tauc = roc_auc_score(tlabels, tprob)

        # ---- val
        model.eval()
        vloss=0.0; vpred=[]; vprob=[]; vlabels=[]
        with torch.no_grad():
            for x,y in tqdm(val_loader, desc="Val"):
                x, y = x.to(device), y.float().unsqueeze(1).to(device)
                out = model(x); loss = criterion(out, y)
                vloss += loss.item()*x.size(0)
                p = torch.sigmoid(out).cpu().numpy().flatten()
                vprob.extend(p); vpred.extend((p>0.5).astype(int)); vlabels.extend(y.cpu().numpy().flatten())

        from sklearn.metrics import roc_auc_score
        acc = accuracy_score(vlabels, vpred); f1 = f1_score(vlabels, vpred)
        auc = roc_auc_score(vlabels, vprob)

        n=len(train_ds); m=len(val_ds)
        history["train_loss"].append(tloss/max(n,1)); history["val_loss"].append(vloss/max(m,1))
        history["train_acc"].append(tacc);  history["val_acc"].append(acc)
        history["train_f1"].append(tf1);    history["val_f1"].append(f1)
        history["train_auc"].append(tauc);  history["val_auc"].append(auc)

        print(f"epoch {epoch}: train loss {history['train_loss'][-1]:.4f} acc {tacc:.3f} f1 {tf1:.3f} auc {tauc:.3f} | "
              f"val loss {history['val_loss'][-1]:.4f} acc {acc:.3f} f1 {f1:.3f} auc {auc:.3f}")

        if history["val_loss"][-1] < best_val_loss:
            best_val_loss = history["val_loss"][-1]
            torch.save({"epoch":epoch, "state_dict":model.state_dict()}, os.path.join(save_dir, "best_model_enet.pth"))
            print("Saved best checkpoint.")

    save_history_json(history, os.path.join(save_dir, "training_history.json"))
    print("Done.")
