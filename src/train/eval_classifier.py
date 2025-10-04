import torch, json, os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from src.utils import device_auto
from src.data.datasets import ImageCSVClassificationDataset
from src.data.transforms import get_val_transform_en
from src.models.efficientnet_heads import EfficientNetBinaryHead

def run_eval(test_csv, checkpoint_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    device = device_auto()

    ds = ImageCSVClassificationDataset(test_csv, transform=get_val_transform_en())
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = EfficientNetBinaryHead().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    y_true=[]; y_prob=[]
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            prob = torch.sigmoid(model(x)).cpu().numpy().flatten()
            y_prob.extend(prob); y_true.extend(y.numpy().flatten())

    y_true = np.array(y_true); y_prob = np.array(y_prob); y_pred = (y_prob>=0.5).astype(int)
    acc = accuracy_score(y_true,y_pred); f1 = f1_score(y_true,y_pred); auc = roc_auc_score(y_true,y_prob)
    print(f"ACC {acc:.4f}  F1 {f1:.4f}  AUC {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5)); plt.imshow(cm, cmap="Blues"); plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.xticks([0,1],["Normal","Anomalous"]); plt.yticks([0,1],["Normal","Anomalous"])
    plt.tight_layout(); out=os.path.join(save_dir,"confusion_matrix.png"); plt.savefig(out,dpi=300); plt.close()
    print("Saved:", out)

    with open(os.path.join(save_dir,"metrics.json"),"w") as f:
        json.dump({"acc":float(acc),"f1":float(f1),"auc":float(auc)}, f)
