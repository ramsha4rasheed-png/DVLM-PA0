# AI 623: Deep Vision Language Models — Assignment 0

A foundational hands-on warm-up covering essential practices in deep vision, representation learning, and multimodal alignment. This repository contains implementations and analysis for four distinct tasks exploring model mechanics, architectures, and latent space dynamics.

---

## Technical Overview

### Task 1: Inner Workings of ResNet-152

* **Feature Transferability & Ablation:** Modified a pre-trained ResNet-152 backbone for CIFAR-10 classification. Investigated feature extraction vs. full fine-tuning trade-offs.


* **Residual Connections:** Evaluated gradient flow and convergence penalties by selectively disabling skip connections in core residual blocks.


* **Representational Hierarchy:** Tracked class separability across early, middle, and late layers using t-SNE and UMAP feature visualizations.



### Task 2: Understanding Vision Transformers (ViT)

* **Attention Map Interpretability:** Extracted class token (`[CLS]`) self-attention weights from the final transformer layer to generate overlaid spatial heatmaps.


* **Robustness & Pooling Analysis:** Evaluated model degradation under random vs. structured patch masking at inference time. Compared performance differentials between linear probes trained on `[CLS]` embeddings versus global average patch pooling.



### Task 3: Training Variational Autoencoders

* **Generative Modeling:** Implemented an encoder-decoder architecture optimized via Evidence Lower Bound (ELBO) on FashionMNIST.


* **Posterior Collapse Mitigation:** Diagnosed and corrected encoder representation collapse (where the encoder matches the prior too closely) by implementing a progressive KL-divergence annealing schedule.



### Task 4: Modality Gap in CLIP

* **Zero-Shot Evaluation:** Audited OpenAI's CLIP model on STL-10 using targeted prompt engineering strategies (plain labels vs. contextual templates).


* **Latent Space Alignment:** Visualized the intrinsic geometric split between vision and language embedding distributions. Applied an Orthogonal Procrustes transformation via SVD to close the modality gap and assessed its downstream impact on classification accuracy.



---

## Setup & Submission Requirements

* **Report Format:** Documented in full ICML format.


* **Code Execution:** PyTorch, `torchvision`, `timm`, `transformers`, `scipy`, and `scikit-learn`.# DVLM-PA0
Analysis on internal mechanism of ResNet-152, Vit, VAEs and CLIP
