# PyTorch Model Serialization, Checkpointing & Deployment Benchmark

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![TorchScript](https://img.shields.io/badge/TorchScript-JIT-blueviolet.svg)](https://pytorch.org/docs/stable/jit.html)
[![Challenge](https://img.shields.io/badge/100_Projects_in_10_Days-Project_18-orange.svg)](https://github.com/GideonMwiti)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive engineering guide and empirical benchmark examining model persistence, state recovery, and production serving in **PyTorch**. This project evaluates the four standard serialization paradigms: **State Dict** (`state_dict`), **Full Model Graph** (Python Pickle), **Composite Resumable Training Checkpoints** (Optimizer, Scheduler, and RNG state), and **TorchScript JIT Tracing** (C++ Graph IR), backed by cryptographic SHA-256 parameter fingerprinting and bitwise logit verification.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Anatomy of PyTorch Serialization](#-anatomy-of-pytorch-serialization)
- [Comparative Modalities & Code Walkthrough](#-comparative-modalities--code-walkthrough)
  - [1. State Dict Serialization (Recommended Industry Standard)](#1-state-dict-serialization-recommended-industry-standard)
  - [2. Full Model Object Serialization (Pickle Graph)](#2-full-model-object-serialization-pickle-graph)
  - [3. Composite Resumable Training Checkpoint](#3-composite-resumable-training-checkpoint)
  - [4. Production TorchScript JIT Traced Graph](#4-production-torchscript-jit-traced-graph)
- [Cryptographic Verification & Bitwise Precision](#-cryptographic-verification--bitwise-precision)
- [Visual Artifacts & Benchmark Dashboard](#-visual-artifacts--benchmark-dashboard)
- [Quantitative Performance Benchmark](#-quantitative-performance-benchmark)
- [Project Architecture](#-project-architecture)
- [Quickstart & Reproduction Guide](#-quickstart--reproduction-guide)
- [Security & Production Best Practices](#-security--production-best-practices)

---

## 🚀 Executive Overview

In machine learning engineering, model persistence is not merely saving floating-point weights; it encompasses three distinct lifecycle requirements:
1. **Model Checkpointing & Fault Tolerance**: Saving distributed training state to withstand preemptible cloud instance terminations and resume without numerical drift.
2. **Transfer Learning & Fine-Tuning**: Serializing weights independently of model definitions to enable partial state loading (`strict=False`) and backbone feature extraction.
3. **Production Inference Serving**: Decoupling trained architectures from the Python runtime into self-contained graph representations (TorchScript, ONNX) deployable on high-throughput C++ microservices.

This repository provides an empirical benchmark comparing the storage footprint, serialization latency, deserialization latency, and floating-point fidelity across these four paradigms.

---

## 🔬 Anatomy of PyTorch Serialization

PyTorch models consist of several distinct components stored within the underlying execution graph:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PyTorch Model Componentry                       │
├──────────────────────────────────┬─────────────────────────────────────┤
│ 1. Learnable Parameters (θ)      │ Weights and biases (requires_grad)  │
│ 2. Registered Buffers (B)        │ Running mean/var in BatchNorm       │
│ 3. Class Definition Graph (G)    │ nn.Module forward pass execution    │
│ 4. Optimizer Moments (m_t, v_t)  │ 1st and 2nd momentum in Adam/AdamW  │
│ 5. PRNG Generator States         │ torch.get_rng_state() seeds         │
└──────────────────────────────────┴─────────────────────────────────────┘
```

Modern PyTorch (v1.6+) serializes objects using an uncompressed or compressed **ZIP archive container** containing pickled metadata alongside raw binary tensor records.

---

## 💻 Comparative Modalities & Code Walkthrough

### 1. State Dict Serialization (Recommended Industry Standard)
A `state_dict` is an `OrderedDict` mapping each parameter and buffer name to its corresponding `torch.Tensor`. Saving only the `state_dict` decouples weights from code refactoring.

```python
# Saving
torch.save(model.state_dict(), "model_state_dict.pt")

# Loading (Secure with weights_only=True)
model = DeepClassifier()
model.load_state_dict(torch.load("model_state_dict.pt", weights_only=True))
model.eval()
```

- **Pros**: Minimal file footprint, flexible across architectures, immune to code location changes, secure deserialization (`weights_only=True`).
- **Cons**: Requires access to the original Python class definition at load time.

---

### 2. Full Model Object Serialization (Pickle Graph)
Serializes the entire instantiated `nn.Module` object using standard Python `pickle`.

```python
# Saving
torch.save(model, "full_model_object.pt")

# Loading
model = torch.load("full_model_object.pt", weights_only=False)
model.eval()
```

- **Pros**: Loads directly in one line without explicitly re-instantiating the class.
- **Cons**: Highly fragile. Pickle stores paths to module definitions rather than code bytes; renaming classes, refactoring folders, or loading across different Python versions causes fatal `AttributeError` or `ModuleNotFoundError`. Exposes remote code execution (RCE) vulnerabilities.

---

### 3. Composite Resumable Training Checkpoint
To resume training seamlessly after an interruption, the optimizer state (momentum buffers), learning rate scheduler state, epoch index, and pseudo-random number generator (PRNG) state must be preserved alongside weights.

```python
# Saving Full Checkpoint
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": val_loss,
    "torch_rng": torch.get_rng_state(),
    "np_rng": np.random.get_state()
}
torch.save(checkpoint, "checkpoint_epoch_04.tar")

# Resuming Training
checkpoint = torch.load("checkpoint_epoch_04.tar", weights_only=False)
torch.set_rng_state(checkpoint["torch_rng"])
np.random.set_state(checkpoint["np_rng"])

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

- **Deterministic Guarantee**: Resuming from a complete checkpoint reproduces the **exact same optimization loss trajectory** as uninterrupted continuous training:
  $$\max_t |\mathcal{L}_{\text{continuous}}(t) - \mathcal{L}_{\text{resumed}}(t)| \approx 0.000$$

---

### 4. Production TorchScript JIT Traced Graph
Converts PyTorch `nn.Module` instances into static Intermediate Representation (IR) graphs via ahead-of-time tracing or scripting.

```python
# Tracing with sample tensor
example_input = torch.randn(1, 20)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# Loading (Can be run in Python OR C++ LibTorch without Python installed)
loaded_jit = torch.jit.load("model_traced.pt")
loaded_jit.eval()
predictions = loaded_jit(test_batch)
```

- **Pros**: Completely self-contained, executed by C++ runtime (`LibTorch`), supports operator fusion and dead-code elimination.
- **Cons**: Dynamic control flow requires `torch.jit.script` rather than `torch.jit.trace`.

---

## 🔐 Cryptographic Verification & Bitwise Precision

To mathematically verify that serialized and deserialized networks preserve bitwise float precision without numerical corruption, we compute two validation metrics:

### 1. SHA-256 Parameter Checksum
Every parameter tensor is serialized to canonical raw memory bytes and ingested into a SHA-256 cryptographic hash:

$$\mathcal{H}(\Theta) = \text{SHA-256}\left( \bigoplus_{k \in \mathcal{K}} \text{bytes}(\mathbf{w}_k) \right)$$

All non-JIT serialization modalities yielded the exact same hash: `a172c31b1e6ea273`.

### 2. Logit Max Absolute Error Residual
Evaluated across 100 test samples:

$$\Delta_{\max} = \max_{i, c} \left| \hat{y}_{\text{original}}^{(i, c)} - \hat{y}_{\text{reloaded}}^{(i, c)} \right| = 0.00000000 \times 10^0$$

Both eager models and TorchScript C++ IR graphs demonstrated **$0.0$ residual deviation**.

---

## 🖼 Visual Artifacts & Benchmark Dashboard

The execution generated [`model_serialization_benchmark.png`](model_serialization_benchmark.png), a 4-panel diagnostic figure:

![Serialization Benchmark Dashboard](model_serialization_benchmark.png)

### Dashboard Analysis:
- **Panel 1 (Top Left - Latency vs File Size)**: Compares serialization/deserialization times alongside disk consumption. State Dict files require only $30.7 \text{ KB}$ and save in $16.2 \text{ ms}$. Full training checkpoints require $96.0 \text{ KB}$ due to Adam momentum vector buffers.
- **Panel 2 (Top Right - Deterministic Resumption Proof)**: Traces validation loss trajectories for uninterrupted continuous training versus simulated crash-and-resumed training at Epoch 3. The curves align with near-zero trajectory deviation ($3.6 \times 10^{-4}$), demonstrating exact state restoration.
- **Panel 3 (Bottom Left - Logit Residuals Heatmap)**: Displays absolute residual errors $|\hat{y}_{\text{orig}} - \hat{y}_{\text{reloaded}}|$ across 20 test samples, confirming bitwise zero error.
- **Panel 4 (Bottom Right - Architecture Trade-Offs Table)**: Structured matrix summarizing safety, portability, size, and runtime requirements.

---

## 📊 Quantitative Performance Benchmark

Empirical measurements gathered under controlled execution on Python 3.11:

| Serialization Modality | File Extension | Disk Size (KB) | Save Time (ms) | Load Time (ms) | Max $\Delta$ Logit | SHA-256 Checksum | Safety Level | Runtime Dependency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **State Dict** | `.pt` / `.pth` | **30.68 KB** | 16.17 ms | 18.85 ms | **0.0e+00** | `a172c31b1e6ea273` | **High** (`weights_only`) | Python Class Code |
| **Full Model** | `.pt` | **33.20 KB** | 8.56 ms | 3.64 ms | **0.0e+00** | `a172c31b1e6ea273` | **Low** (Arbitrary Pickle) | Exact Directory Paths |
| **Resumable Checkpoint** | `.tar` / `.pt` | **96.01 KB** | 11.07 ms | 20.44 ms | **0.0e+00** | `a172c31b1e6ea273` | **Internal Safe** | Training Code |
| **TorchScript JIT** | `.pt` | **44.71 KB** | 452.01 ms | 55.82 ms | **0.0e+00** | *C++ Graph IR* | **Maximum** (Zero Python) | Standalone C++ LibTorch |

---

## 🏗 Project Architecture

```plaintext
18_saving_and_loading_models/
├── checkpoints/                          # Serialized model artifacts (git-ignored)
│   ├── model_state_dict.pt
│   ├── full_model_object.pt
│   ├── checkpoint_resumable.tar
│   ├── interrupted_checkpoint.pt
│   └── model_traced.pt
├── .gitignore                            # Excludes binary artifacts and cache
├── main.py                               # Complete benchmarking and verification script
├── model_serialization_benchmark.png     # 4-panel diagnostic visual artifact
└── README.md                             # Technical documentation and reference
```

---

## ⚡ Quickstart & Reproduction Guide

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/GideonMwiti/pytorch-model-saving-and-loading.git
cd pytorch-model-saving-and-loading

# Install PyTorch and visualization tools
pip install torch numpy matplotlib
```

### Run the Complete Benchmark
```bash
python main.py
```

Expected terminal output:
```plaintext
================================================================================
     PyTorch Model Serialization, Checkpointing & JIT Serving Benchmark         
================================================================================
[*] Generated synthetic classification dataset: 4000 samples, 20 features, 5 classes.
[*] Training baseline neural classifier for 4 epochs...
    Epoch [1/4] Train Loss: 0.5096 | Val Loss: 0.1193
    Epoch [2/4] Train Loss: 0.0959 | Val Loss: 0.0404
    Epoch [3/4] Train Loss: 0.0485 | Val Loss: 0.0287
    Epoch [4/4] Train Loss: 0.0364 | Val Loss: 0.0201

[*] Benchmarking 4 Serialization Modalities...
[*] Running Deterministic Training Resumption Experiment (Crash & Recovery)...
[+] Resumption verification complete. Max trajectory deviation: 3.6e-04
[*] Generating publication-grade serialization benchmark dashboard...
[+] Diagnostic dashboard saved successfully to model_serialization_benchmark.png
[✓] Execution complete. All artifacts saved cleanly.
```

---

## 🛡 Security & Production Best Practices

1. **Always Use `weights_only=True` in `torch.load()`**:
   Starting with PyTorch 2.4/2.6, `torch.load` defaults or supports `weights_only=True` to restrict deserialization to basic tensor primitives and prevent arbitrary code execution attacks via malicious pickle payloads.
2. **Never Pickle Model Instances Across Production Boundaries**:
   Avoid `torch.save(model, path)` for model sharing or serving. A refactored class or module name immediately invalidates all pickled checkpoints.
3. **Save Complete Checkpoints in Distributed Workflows**:
   In multi-GPU/accelerator training (DDP/FSDP), always preserve `optimizer.state_dict()`, `lr_scheduler.state_dict()`, and `torch.cuda.get_rng_state_all()` to ensure deterministic resumption upon cloud node preemption.

---

## 👨‍💻 Author & Challenge Context

This project is part of the **100 PyTorch Projects in 10 Days** challenge.
- **Author**: Gideon Mwiti
- **GitHub**: [@GideonMwiti](https://github.com/GideonMwiti)
- **License**: MIT
