import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetBinaryHead(nn.Module):
    """
    EfficientNet-B0 backbone (pretrained) + custom MLP head.
    Outputs a single logit for binary classification (BCEWithLogitsLoss).
    """
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b0')
        for p in self.base.parameters():
            p.requires_grad = False  # warmup freeze
        in_feats = self.base._fc.in_features
        self.base._fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        f = self.base(x)
        return self.classifier(f)

def unfreeze_schedule(model: 'EfficientNetBinaryHead', epoch: int):
    # staged unfreezing epochs 3/8/15
    if epoch == 3:
        for p in model.classifier.parameters(): p.requires_grad = True
    elif epoch == 8:
        for name, p in model.base.named_parameters():
            if 'blocks.5' in name or 'blocks.6' in name or 'blocks.7' in name:
                p.requires_grad = True
    elif epoch == 15:
        for p in model.parameters(): p.requires_grad = True
