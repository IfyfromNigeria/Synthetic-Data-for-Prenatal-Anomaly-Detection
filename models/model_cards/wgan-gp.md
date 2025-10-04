# Model Card: WGAN-GP Generator

**Task:** Conditional image synthesis (NT ROI)  
**Conditioning:** Class label (normal vs anomalous)  
**Training objective:** Wasserstein GAN with gradient penalty (λ=10, 5 critic updates per generator step)  

**Notes:** Generates realistic NT textures without strong mode collapse.  

**Usage Example:**
```bash
python -m src.generate_synthetic --num 64 --class anomalous \
  --checkpoint models/checkpoints/wgan_gp_nt_conditional.pt \
  --out outputs/samples/
