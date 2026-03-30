# 🌫️ DDPM from Scratch — Denoising Diffusion Probabilistic Model on MNIST

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![Colab](https://img.shields.io/badge/Run%20on-Google%20Colab-yellow?logo=googlecolab)](https://colab.research.google.com)

A clean, minimal implementation of **Denoising Diffusion Probabilistic Models (DDPM)** built from scratch in PyTorch, trained on the MNIST handwritten digits dataset. This project walks through every component of a diffusion model — from the forward noising process to the reverse denoising U-Net — making it ideal for learning and experimentation.

---

## 🧠 What is DDPM?

Diffusion models are a class of generative models that learn to **reverse a gradual noising process**. During training:

1. **Forward process**: Gaussian noise is incrementally added to an image over `T` timesteps until it becomes pure noise.
2. **Reverse process**: A neural network learns to predict and remove the noise step-by-step, eventually reconstructing a clean image from random noise.

This implementation follows the original paper:
> **[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)** — Ho et al., NeurIPS 2020

---

## ✨ Features

- ✅ **Linear noise schedule** with configurable `β_start` and `β_end`
- ✅ **Forward diffusion** — visualizes noising at multiple timesteps
- ✅ **Sinusoidal time embeddings** — injects timestep information into the network
- ✅ **U-Net denoiser** with residual blocks, attention, and time conditioning
- ✅ **DDPM reverse sampling** — generates images from pure Gaussian noise
- ✅ **Trained and tested on MNIST** — simple, fast, and reproducible
- ✅ **Google Colab compatible** — runs end-to-end in a free GPU runtime

---


---

## 🚀 Quickstart

### ▶️ Run on Google Colab (Recommended)

Click the badge above or open `Diffusion_Model.ipynb` directly in Colab. All dependencies are pre-installed.

### 💻 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ddpm-mnist-from-scratch.git
cd ddpm-mnist-from-scratch

# 2. Install dependencies
pip install torch torchvision matplotlib

# 3. Launch the notebook
jupyter notebook Diffusion_Model.ipynb
```

---

## 🔬 Implementation Details

### Noise Schedule

A **linear beta schedule** is used with `T = 1000` timesteps:

```python
T = 1000
beta_start = 1e-4
beta_end   = 0.02
betas      = torch.linspace(beta_start, beta_end, T)
```

The cumulative product `ᾱ_t = ∏ αₜ` allows direct noising at any timestep without sequential computation.

### Forward Process

Given a clean image `x₀`, the noisy image at timestep `t` is computed in closed form:

```
xₜ = √(ᾱ_t) · x₀  +  √(1 - ᾱ_t) · ε,    ε ~ N(0, I)
```

```python
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)
    sqrt_alpha_bar     = torch.sqrt(alpha_bar[t])[:, None, None, None]
    sqrt_one_minus_ab  = torch.sqrt(1 - alpha_bar[t])[:, None, None, None]
    return sqrt_alpha_bar * x0 + sqrt_one_minus_ab * noise, noise
```

### Sinusoidal Time Embedding

Timestep `t` is encoded using sinusoidal positional embeddings (similar to transformers) and projected through a small MLP:

```python
class SinusoidalTimeEmbedding(nn.Module):
    def forward(self, t):
        half  = self.emb_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        args  = t[:, None].float() * freqs[None]
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)
```

### Reverse Process (Sampling)

Starting from `x_T ~ N(0, I)`, the model iteratively denoises using the learned noise prediction:

```python
@torch.no_grad()
def reverse_step(model, xt, t_val, alphas, alpha_bar):
    pred_noise = model(xt, t)
    mean = (1 / sqrt(α_t)) * (xₜ - ((1 - α_t) / √(1 - ᾱ_t)) * pred_noise)
    if t_val > 0:
        sigma = sqrt(1 - α_t)
        return mean + sigma * torch.randn_like(xt)
    return mean
```

---

## 📊 Results

### Forward Noising Process

Visualizing a digit being progressively destroyed by noise across timesteps:

| t=0 | t=50 | t=100 | t=150 | t=200 | t=250 | t=999 |
|-----|------|-------|-------|-------|-------|-------|
| Clean | Slight | Moderate | Heavy | Very Heavy | Almost noise | Pure noise |

### Generated Samples (after training)

After 1 epoch of training on MNIST, the model generates recognizable handwritten-style digits from pure Gaussian noise.

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Timesteps `T` | 1000 |
| `β_start` | 1e-4 |
| `β_end` | 0.02 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Grad clip norm | 1.0 |
| Dataset | MNIST (28×28, grayscale) |
| Image range | [-1, 1] |

---




## 🔮 Future Improvements

- [ ] **More epochs & deeper U-Net** — train longer with a larger model for sharper, more diverse samples
- [ ] **Cosine noise schedule** — replace linear β schedule with cosine (as proposed by Nichol & Dhariwal 2021) for better sample quality
- [ ] **Classifier-free guidance** — add conditional generation so the model can target specific digits (0–9)
- [ ] **Scale to CIFAR-10 / CelebA** — move beyond MNIST to color images with a more powerful backbone
- [ ] **Attention in U-Net** — add self-attention layers at lower resolutions for better global coherence




## 📚 References

- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). *NeurIPS 2020*.


## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a PR or issue.

---


> Built with ❤️ using PyTorch. If this helped you learn diffusion models, give it a ⭐!
