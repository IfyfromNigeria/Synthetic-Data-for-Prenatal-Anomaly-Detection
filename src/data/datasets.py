import cv2, torch
from torch.utils.data import Dataset
import pandas as pd

def apply_clahe(gray, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)

class ImageCSVClassificationDataset(Dataset):
    """
    CSV columns: image_path, label
    Applies: resize(224), CLAHE, convert to RGB, Albumentations transform.
    """
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = row['image_path']
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Image not found: {p}")
        img = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = apply_clahe(gray)
        rgb = cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)
        if self.transform:
            x = self.transform(image=rgb)['image']
        else:
            x = torch.tensor(rgb).permute(2,0,1).float()/255.0
        y = torch.tensor(row['label'], dtype=torch.float32)
        return x, y
