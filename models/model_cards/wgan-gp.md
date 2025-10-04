# Model Card: WGAN-GP Generator

**Task:** Conditional image synthesis (NT ROI)  
**Conditioning:** Class label (normal vs anomalous)  
**Training objective:** Wasserstein GAN with gradient penalty (λ=10, 5 critic updates per generator step)  

**Notes:** Generates realistic NT textures without strong mode collapse.  

**Download weights: [https://github.com/IfyfromNigeria/Synthetic-Data-for-Prenatal-Anomaly-Detection/releases/download/v1.0.0/wgan_gp_checkpoint.pt]

**SHA256 checksum:**  
17c526a2e10fda4803ebaa896dca8686c675638edfe815606359e820c58f28be

**Usage Example:**
```bash
python -m src.generate_synthetic --num 64 --class anomalous \
  --checkpoint models/checkpoints/wgan_gp_nt_conditional.pt \
  --out outputs/samples/
