import torch, cv2, os, numpy as np, matplotlib.pyplot as plt
from src.models.efficientnet_heads import EfficientNetBinaryHead
from src.data.transforms import get_val_transform_en

class EfficientNetGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self.target_layer = self.model.base._blocks[-1]
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output): self.activations = output
    def _save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.eval(); self.model.zero_grad()
        out = self.model(x)
        out.backward()
        A = self.activations.detach(); G = self.gradients.detach()
        w = torch.mean(G, dim=[2,3], keepdim=True)
        cam = torch.sum(w * A, dim=1, keepdim=True).relu()
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max()>0 else torch.zeros_like(cam)
        return cam.squeeze().cpu().numpy()

def visualize_gradcam_on_images(model_ckpt, image_paths, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EfficientNetBinaryHead().to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device)["state_dict"])
    cammer = EfficientNetGradCAM(model)
    tfm = get_val_transform_en()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    figs = []
    for p in image_paths:
        img = cv2.imread(p); img = cv2.resize(img, (224,224))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        x = tfm(image=rgb)['image'].unsqueeze(0).to(device).requires_grad_(True)
        cam = cammer(x)
        cam = cv2.resize(cam, (224,224))
        heat = (cam*255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        over = cv2.addWeighted(rgb, 0.6, heat, 0.4, 0)
        figs.append(over[:,:,::-1])

    cols = min(5,len(figs)); rows = (len(figs)+cols-1)//cols
    plt.figure(figsize=(4*cols,4*rows))
    for i,im in enumerate(figs):
        plt.subplot(rows,cols,i+1); plt.imshow(im); plt.axis('off')
    plt.tight_layout(); plt.savefig(save_path,dpi=300); plt.close()
    print("Saved Grad-CAM grid to:", save_path)
