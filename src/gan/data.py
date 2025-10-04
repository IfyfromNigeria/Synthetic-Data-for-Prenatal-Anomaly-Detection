import os, cv2, torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class WGANDataset(Dataset):
    def __init__(self, csv_path, img_size=64):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['structure'].fillna('').str.upper().eq('NT')] if 'structure' in self.df.columns else self.df
        self.img_size = img_size

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        p = str(r['image_path'])
        label = int(r['label'])
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return torch.zeros((1,self.img_size,self.img_size)), torch.tensor(-1)
        H,W = img.shape[:2]
        hmin = int(np.clip(round(r.get('h_min',0)),0,H-1))
        wmin = int(np.clip(round(r.get('w_min',0)),0,W-1))
        hmax = int(np.clip(round(r.get('h_max',H)),hmin+1,H))
        wmax = int(np.clip(round(r.get('w_max',W)),wmin+1,W))
        crop = img[hmin:hmax, wmin:wmax]
        if crop.size==0:
            return torch.zeros((1,self.img_size,self.img_size)), torch.tensor(-1)
        crop = cv2.resize(crop, (self.img_size, self.img_size))
        x = torch.tensor(crop, dtype=torch.float32).unsqueeze(0)
        x = (x/127.5) - 1.0
        y = torch.tensor(label, dtype=torch.long)
        return x, y
