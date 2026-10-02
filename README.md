# Marine Debris Detection from Sentinel-2 Imagery

Semantic segmentation of floating marine debris in Sentinel-2 satellite imagery on the MARIDA benchmark. We propose **DeepSpectralUNet**, an encoder-decoder that processes four spectral indices (FDI, NDVI, NDWI, FAI) in a dedicated multi-scale branch and fuses them into every decoder stage, and compare it against classical and deep-learning baselines.

The full write-up is in [`marine_debris_report.pdf`](marine_debris_report.pdf) (*Marine Debris Detection from Sentinel-2 Imagery via Multi-Scale Spectral Index Fusion*).

## Task
Per-pixel classification of 256 × 256 patches (11 spectral bands) into 10 classes: Marine Debris, Sargassum, Natural Organic Material, Ship, Clouds, Water, Foam, Waves, Cloud Shadows, Wakes. Debris is only 0.45% of labelled pixels, so the training setup uses inverse-frequency loss weights, debris-aware patch sampling, a warmup + cosine-annealing learning-rate schedule and gradient clipping. MARIDA splits: 694 train / 328 val / 359 test patches.

## Test-set results
| Model | Debris IoU | Debris F1 | Macro F1 | Accuracy |
|-------|-----------:|----------:|---------:|---------:|
| Random Forest | 0.6725 | 0.8042 | 0.5620 | 0.9235 |
| U-Net | 0.1900 | 0.3193 | 0.5113 | 0.8876 |
| Attention U-Net | 0.1996 | 0.3327 | 0.5074 | 0.8728 |
| DeepLabV3+ | 0.0671 | 0.1258 | 0.3631 | 0.7179 |
| ResAttUNet-CBAM | 0.4205 | 0.5921 | 0.7477 | 0.9548 |
| **DeepSpectralUNet (proposed)** | **0.4264** | **0.5978** | 0.6190 | 0.9350 |

DeepSpectralUNet gets the best debris IoU and F1 among the deep-learning models. The Random Forest baseline still has the highest debris IoU/F1 overall, and ResAttUNet-CBAM has the best macro F1 and accuracy. The report discusses these trade-offs and the limits imposed by the very small number of debris pixels.

## Repository contents
| File | Description |
|------|-------------|
| `marine_debris.ipynb` | Main notebook: data loading, models, training, evaluation |
| `marine_debris-checkpoint.ipynb`, `saideep-checkpoint.ipynb` | Additional notebook copies (checkpoint versions of the main notebook) |
| `marine_debris_report.pdf` | Project report |
| `requirements.txt` | Python dependencies (tested on Python 3.13, CUDA 12.8, PyTorch 2.12) |

## Setup
```bash
pip install -r requirements.txt
```
Download MARIDA and set the `DATA_ROOT` path at the top of the notebook (the committed notebooks contain the author's local paths). A CUDA GPU is recommended. The notebook can either retrain the models or load saved checkpoints; checkpoints are not included in this repository.

## Team
Blekinge Institute of Technology, Karlskrona, Sweden.
- **Sriya Chittaneni**: design and implementation of DeepSpectralUNet (multi-scale spectral branch, attention mechanisms, residual encoder) and part of the training pipeline
- **Meghashyam Sai Dontha**: preprocessing, spectral feature engineering, class-imbalance handling, augmentation, evaluation and results analysis
- **Saideep Reddy Tamma**: baseline models, hyperparameter tuning, visualisations

## Reference
K. Kikaki, I. Kakogeorgiou, P. Mikeli, D. E. Raitsos, K. Karantzalos, "MARIDA: A benchmark for marine debris detection from Sentinel-2 remote sensing data," *PLOS ONE* 17(1), e0262247, 2022.
