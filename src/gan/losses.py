import torch

def gradient_penalty(critic, real, fake, onehot_labels, device):
    b = real.size(0)
    alpha = torch.rand(b,1,1,1, device=device)
    inter = alpha*real + (1-alpha)*fake
    inter.requires_grad_(True)
    scores = critic(inter, onehot_labels)
    grad = torch.autograd.grad(outputs=scores, inputs=inter,
                               grad_outputs=torch.ones_like(scores),
                               create_graph=True, retain_graph=True)[0]
    grad = grad.view(b, -1)
    gp = ((grad.norm(2, dim=1) - 1.0)**2).mean()
    return gp
