# Deep Pensieve™

### A Deep Residual Enhanced Attention Maximizing Super Resolution Variational Auto-Encoder (DREAM+SRVAE)

Deep Pensieve is a multi-stage variational autoencoder architecture designed for high-quality image generation and super-resolution. It combines several state-of-the-art techniques to overcome the traditional blurriness problem of VAEs while enabling coarse-to-fine image synthesis.

---

## Overview

Traditional VAEs generate blurry images due to Gaussian assumptions and L2 loss constraints. Deep Pensieve addresses this through:

- **Multi-stage generation** — Coarse-to-fine refinement pipeline
- **Deep residual networks** — Enhanced capacity with skip connections
- **Attention mechanisms** — Channel and spatial attention for adaptive feature refinement
- **Perceptual loss** — DSSIM-based loss for sharper reconstructions
- **Sub-pixel super-resolution** — Efficient upscaling in feature space

## Architecture

```
Input → Encoder → Latent Space → Decoder → Refiner → Super-Resolution → Output
         ↓                         ↓          ↓              ↓
      64→4×4                    4×4→64      64×64        64→128×128
```

### Encoder
- Stacked 3×3 convolutions with dilated convolutions for expanded receptive field
- Group Normalization for batch-size independent training
- Progressive downsampling to 4×4 feature maps
- Projects to configurable latent dimension (default: 512)

### Decoder
- Sub-pixel upscaling (no checkerboard artifacts)
- Symmetric architecture to encoder
- Dilated convolutions for context aggregation

### Refiner
- Deep residual network based on EDSR architecture
- Residual groups with residual blocks
- Bottleneck Attention Modules (channel + spatial attention)
- Global skip connection for stable training

### Super-Resolution Module
- Sub-pixel convolution for efficient 2× upscaling
- Operates in low-resolution feature space
- Real-time capable architecture

## Key Techniques

| Component | Technique | Reference |
|-----------|-----------|-----------|
| Normalization | Group Normalization | [Wu & He, 2018](https://arxiv.org/abs/1803.08494) |
| Latent Loss | Maximum Mean Discrepancy (MMD) | [InfoVAE, 2017](https://arxiv.org/abs/1706.02262) |
| Reconstruction Loss | DSSIM (Structural Similarity) | [Snell et al., 2015](https://arxiv.org/abs/1511.06409) |
| Upsampling | Sub-Pixel Convolution | [Shi et al., 2016](https://arxiv.org/abs/1609.05158) |
| Residual Blocks | EDSR-style (no BatchNorm) | [Lim et al., 2017](https://arxiv.org/abs/1707.02921) |
| Attention | Channel + Spatial (BAM/RCAB) | [Park et al., 2018](https://arxiv.org/abs/1807.06514) |
| Dilated Conv | DRN for resolution preservation | [Yu et al., 2017](https://arxiv.org/abs/1705.09914) |

## Installation

### Requirements
- Python 3.7+
- TensorFlow 1.x / Keras
- keras-contrib

```bash
pip install tensorflow-gpu keras keras-contrib numpy
```

### Project Structure
```
deep-pensieve/
├── deep_pensieve.ipynb    # Main notebook
├── libs/
│   ├── utils.py           # Image loading utilities
│   ├── gif.py             # GIF generation
│   └── variance_pooling.py # Custom pooling layer
└── imgs/
    └── [dataset]/         # Training images
```

## Usage

### Configuration

```python
# Image settings
DIRECTORY = 'your_dataset'
SIZE = 64                    # Base resolution (32, 64, 128, 256)
SCALE_FACTOR = 2             # Super-resolution factor

# Architecture
LATENT_DIM = 512
FILTERS = [64, 128, 192, 256]  # Encoder/decoder channels

# Refiner
R_GROUPS = 4                 # Residual groups
R_BLOCKS = 4                 # Blocks per group
R_FILTERS = 64               # Channel width
R_ATTENTION = True           # Enable attention

# Training
BATCH_SIZE = 8
MULTI_GPU = True
```

### Training

```python
# Load and normalize images
imgs, xs, ys = utils.load_images(directory="imgs/"+DIRECTORY, rx=SIZE, ry=SIZE)
IMGS = imgs / 127.5 - 1

# Train
MODEL.fit(
    x=IMGS_F,
    y=[IMGS_F, IMGS_C, IMGS_F, IMGS_C, IMGS_F, IMGS_2X_C, IMGS_2X_F],
    batch_size=BATCH_SIZE,
    epochs=100,
    callbacks=[giffer, saver]
)
```

### Generation

```python
# Random latent interpolation
random_latents(n_imgs=3, steps=30)

# Encode → Decode pipeline
latent = ENCODER.predict(image)
reconstruction = DECODER.predict(latent)
super_res = SUPERSIZER.predict(reconstruction)
```

### Latent Space Interpolation

The model supports smooth interpolation between images in latent space:

```python
def latent_animation(imgs, steps=30):
    latents = [ENCODER.predict(img) for img in imgs]
    # Interpolate between consecutive latent vectors
    path = interpolate(latents, steps)
    # Generate frames
    frames = DECODER.predict(path)
    return SUPERSIZER.predict(frames)
```

## Model Outputs

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| `ENCODER` | Image | Latent vector | Feature extraction |
| `DECODER` | Latent vector | 64×64 image | Generation |
| `SUPER` | Latent vector | 128×128 image | Direct super-res generation |
| `SUPERSIZER` | 64×64 image | 128×128 image | Image upscaling |
| `AUTOENCODER` | Image | All intermediate outputs | Training |

## Loss Functions

The model uses a composite loss:

1. **MMD Loss** — Matches latent distribution to unit Gaussian
2. **DSSIM Loss** — Perceptual similarity for decoder and refiner outputs
3. **MSE/MAE** — Pixel-level reconstruction

```python
LOSSES = {
    'encoder': vae_loss,           # MMD
    'decoder': 'mae',              # Reconstruction
    'decoder_dssim': DSSIMObjective(),
    'refiner': 'mae',
    'refiner_dssim': DSSIMObjective(),
    'super': 'mae',
    'super_dssim': DSSIMObjective()
}
```

## References

- [Multi-Stage VAE](https://arxiv.org/abs/1705.07202) — Coarse-to-fine generation
- [InfoVAE](https://arxiv.org/abs/1706.02262) — MMD-based latent regularization  
- [EDSR](https://arxiv.org/abs/1707.02921) — Enhanced deep residual SR
- [RCAN](https://arxiv.org/abs/1807.02758) — Residual channel attention
- [BAM](https://arxiv.org/abs/1807.06514) — Bottleneck attention module
- [Sub-Pixel CNN](https://arxiv.org/abs/1609.05158) — Efficient upscaling
- [Deconv Checkerboard](https://distill.pub/2016/deconv-checkerboard/) — Artifact-free upsampling

## License

MIT License

---

*Deep Pensieve — Extracting memories from the latent space*
