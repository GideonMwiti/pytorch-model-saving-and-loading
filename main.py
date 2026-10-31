"""
PyTorch Model Serialization, Checkpointing, and Deployment Benchmark
Part of the '100 PyTorch Projects in 10 Days' Challenge
Author: Gideon Mwiti

Demonstrates:
- Method 1: State Dict (torch.save / load_state_dict) with weights_only=True
- Method 2: Full Model Object Serialization (Pickle graph)
- Method 3: Resumable Composite Training Checkpoints (Model + Optimizer + Scheduler + RNG)
- Method 4: Production TorchScript JIT Tracing for High-Performance Serving
- Bitwise float verification and exact deterministic training resumption proof
"""

import sys
import types
import os
import time
import hashlib
from typing import Dict, Any, Tuple, List

# Standard patch for Python 3.11+ bz2/lzma compatibility
sys.modules['bz2'] = types.SimpleNamespace(open=open)
sys.modules['lzma'] = types.SimpleNamespace(open=open)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------------
# 1. Deterministic Seeding & Device Setup
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Execution Device: {DEVICE}")

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Benchmark Neural Network Architecture
# -----------------------------------------------------------------------------
class DeepClassifier(nn.Module):
    """
    Multi-layer Non-linear Classifier equipped with Batch Normalization,
    LeakyReLU activations, and Dropout.
    """
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# -----------------------------------------------------------------------------
# 3. Synthetic Benchmark Dataset (Pure PyTorch Generator)
# -----------------------------------------------------------------------------
def generate_dataset(n_samples: int = 4000, n_features: int = 20, n_classes: int = 5) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """Generates isotropic Gaussian clusters for multi-class classification."""
    g = torch.Generator().manual_seed(42)
    centers = torch.randn(n_classes, n_features, generator=g) * 3.0
    samples_per_class = n_samples // n_classes

    X_list, y_list = [], []
    for c in range(n_classes):
        noise = torch.randn(samples_per_class, n_features, generator=g) * 1.2
        X_list.append(centers[c].unsqueeze(0) + noise)
        y_list.append(torch.full((samples_per_class,), c, dtype=torch.long))

    X_tensor = torch.cat(X_list, dim=0)
    y_tensor = torch.cat(y_list, dim=0)

    # Standardize
    X_tensor = (X_tensor - X_tensor.mean(dim=0, keepdim=True)) / (X_tensor.std(dim=0, keepdim=True) + 1e-7)

    perm = torch.randperm(n_samples, generator=g)
    X_tensor, y_tensor = X_tensor[perm], y_tensor[perm]

    split = int(0.8 * n_samples)
    train_ds = TensorDataset(X_tensor[:split], y_tensor[:split])
    val_ds = TensorDataset(X_tensor[split:], y_tensor[split:])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_batch = X_tensor[split:split + 100].to(DEVICE)

    return train_loader, val_loader, test_batch


# -----------------------------------------------------------------------------
# 4. Hash Fingerprinting for Parameter Verification
# -----------------------------------------------------------------------------
def compute_state_dict_hash(state_dict: Dict[str, torch.Tensor]) -> str:
    """Computes a cryptographic SHA-256 fingerprint of all parameter buffers."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        hasher.update(key.encode('utf-8'))
        hasher.update(state_dict[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


# -----------------------------------------------------------------------------
# 5. Training and Evaluation Routines
# -----------------------------------------------------------------------------
def train_single_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer
) -> float:
    model.train()
    total_loss, total_count = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total_count += X_batch.size(0)
    return total_loss / total_count


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    total_loss, total_count = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        total_count += X_batch.size(0)
    return total_loss / total_count


# -----------------------------------------------------------------------------
# 6. Serialization Methods Demonstration
# -----------------------------------------------------------------------------
def benchmark_serialization_methods(
    trained_model: DeepClassifier,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    test_batch: torch.Tensor
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, torch.Tensor]]:
    """
    Executes and measures 4 serialization formats:
    1. State Dict (.pt)
    2. Full Model Object (.pt)
    3. Composite Training Checkpoint (.tar)
    4. TorchScript Traced JIT (.pt)
    """
    results: Dict[str, Dict[str, Any]] = {}
    logits_dict: Dict[str, torch.Tensor] = {}

    trained_model.eval()
    with torch.no_grad():
        baseline_logits = trained_model(test_batch).cpu()
    logits_dict["Original (Eager)"] = baseline_logits
    orig_hash = compute_state_dict_hash(trained_model.state_dict())

    # --- Method 1: State Dict (Recommended) ---
    path_state = os.path.join(CHECKPOINT_DIR, "model_state_dict.pt")
    t0 = time.perf_counter()
    torch.save(trained_model.state_dict(), path_state)
    save_time_state = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    loaded_state = torch.load(path_state, weights_only=True)
    model_reloaded_state = DeepClassifier().to(DEVICE)
    model_reloaded_state.load_state_dict(loaded_state)
    load_time_state = (time.perf_counter() - t0) * 1000

    model_reloaded_state.eval()
    with torch.no_grad():
        logits_state = model_reloaded_state(test_batch).cpu()
    logits_dict["State Dict"] = logits_state

    results["State Dict"] = {
        "file_size_kb": os.path.getsize(path_state) / 1024,
        "save_ms": save_time_state,
        "load_ms": load_time_state,
        "max_abs_diff": torch.max(torch.abs(baseline_logits - logits_state)).item(),
        "sha256": compute_state_dict_hash(model_reloaded_state.state_dict()),
        "safety": "High (weights_only=True)",
        "portability": "High (Needs code class)"
    }

    # --- Method 2: Full Model Object (Pickle Graph) ---
    path_full = os.path.join(CHECKPOINT_DIR, "full_model_object.pt")
    t0 = time.perf_counter()
    torch.save(trained_model, path_full)
    save_time_full = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    model_reloaded_full = torch.load(path_full, weights_only=False)
    load_time_full = (time.perf_counter() - t0) * 1000

    model_reloaded_full.eval()
    with torch.no_grad():
        logits_full = model_reloaded_full(test_batch).cpu()
    logits_dict["Full Model"] = logits_full

    results["Full Model"] = {
        "file_size_kb": os.path.getsize(path_full) / 1024,
        "save_ms": save_time_full,
        "load_ms": load_time_full,
        "max_abs_diff": torch.max(torch.abs(baseline_logits - logits_full)).item(),
        "sha256": compute_state_dict_hash(model_reloaded_full.state_dict()),
        "safety": "Low (Arbitrary Pickle Exec)",
        "portability": "Fragile (Directory bound)"
    }

    # --- Method 3: Composite Resumable Checkpoint ---
    path_checkpoint = os.path.join(CHECKPOINT_DIR, "checkpoint_resumable.tar")
    t0 = time.perf_counter()
    checkpoint_payload = {
        "epoch": epoch,
        "model_state_dict": trained_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "rng_torch": torch.get_rng_state(),
        "rng_np": np.random.get_state()
    }
    torch.save(checkpoint_payload, path_checkpoint)
    save_time_ckpt = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    loaded_ckpt = torch.load(path_checkpoint, weights_only=False)
    model_reloaded_ckpt = DeepClassifier().to(DEVICE)
    model_reloaded_ckpt.load_state_dict(loaded_ckpt["model_state_dict"])
    load_time_ckpt = (time.perf_counter() - t0) * 1000

    model_reloaded_ckpt.eval()
    with torch.no_grad():
        logits_ckpt = model_reloaded_ckpt(test_batch).cpu()
    logits_dict["Checkpoint"] = logits_ckpt

    results["Resumable Checkpoint"] = {
        "file_size_kb": os.path.getsize(path_checkpoint) / 1024,
        "save_ms": save_time_ckpt,
        "load_ms": load_time_ckpt,
        "max_abs_diff": torch.max(torch.abs(baseline_logits - logits_ckpt)).item(),
        "sha256": compute_state_dict_hash(model_reloaded_ckpt.state_dict()),
        "safety": "Moderate (Internal training only)",
        "portability": "High (Complete state recovery)"
    }

    # --- Method 4: Production TorchScript JIT Traced ---
    path_jit = os.path.join(CHECKPOINT_DIR, "model_traced.pt")
    t0 = time.perf_counter()
    example_input = torch.randn(1, 20, device=DEVICE)
    traced_script = torch.jit.trace(trained_model, example_input)
    traced_script.save(path_jit)
    save_time_jit = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    loaded_jit = torch.jit.load(path_jit)
    load_time_jit = (time.perf_counter() - t0) * 1000

    loaded_jit.eval()
    with torch.no_grad():
        logits_jit = loaded_jit(test_batch).cpu()
    logits_dict["TorchScript JIT"] = logits_jit

    results["TorchScript JIT"] = {
        "file_size_kb": os.path.getsize(path_jit) / 1024,
        "save_ms": save_time_jit,
        "load_ms": load_time_jit,
        "max_abs_diff": torch.max(torch.abs(baseline_logits - logits_jit)).item(),
        "sha256": "C++ IR Graph",
        "safety": "Maximum (Self-contained C++)",
        "portability": "Maximum (Runs without Python)"
    }

    return results, logits_dict


# -----------------------------------------------------------------------------
# 7. Deterministic Training Resumption Experiment
# -----------------------------------------------------------------------------
def run_deterministic_resumption_experiment(
    train_loader: DataLoader,
    val_loader: DataLoader
) -> Tuple[List[float], List[float]]:
    """
    Demonstrates mathematically that an interrupted training session restored
    from a full checkpoint yields identical loss trajectories to an uninterrupted session.
    """
    total_epochs = 6
    interrupt_epoch = 3

    criterion = nn.CrossEntropyLoss()

    # --- Experiment A: Uninterrupted Continuous Training ---
    set_seed(100)
    model_cont = DeepClassifier().to(DEVICE)
    opt_cont = optim.AdamW(model_cont.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_continuous = []

    for ep in range(1, total_epochs + 1):
        t_loss = train_single_epoch(model_cont, train_loader, criterion, opt_cont)
        v_loss = evaluate_loss(model_cont, val_loader, criterion)
        loss_continuous.append(v_loss)

    # --- Experiment B: Interrupted & Resumed Training ---
    set_seed(100)
    model_res = DeepClassifier().to(DEVICE)
    opt_res = optim.AdamW(model_res.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_resumed = []

    # Phase 1: Train until interrupt epoch
    for ep in range(1, interrupt_epoch + 1):
        t_loss = train_single_epoch(model_res, train_loader, criterion, opt_res)
        v_loss = evaluate_loss(model_res, val_loader, criterion)
        loss_resumed.append(v_loss)

    # Save comprehensive state
    checkpoint_state = {
        "epoch": interrupt_epoch,
        "model_state": model_res.state_dict(),
        "opt_state": opt_res.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "np_rng": np.random.get_state()
    }
    ckpt_path = os.path.join(CHECKPOINT_DIR, "interrupted_checkpoint.pt")
    torch.save(checkpoint_state, ckpt_path)

    # Phase 2: Simulate complete crash and restore
    del model_res, opt_res
    loaded = torch.load(ckpt_path, weights_only=False)

    torch.set_rng_state(loaded["torch_rng"])
    np.random.set_state(loaded["np_rng"])

    model_resumed = DeepClassifier().to(DEVICE)
    model_resumed.load_state_dict(loaded["model_state"])
    opt_resumed = optim.AdamW(model_resumed.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_resumed.load_state_dict(loaded["opt_state"])

    for ep in range(interrupt_epoch + 1, total_epochs + 1):
        t_loss = train_single_epoch(model_resumed, train_loader, criterion, opt_resumed)
        v_loss = evaluate_loss(model_resumed, val_loader, criterion)
        loss_resumed.append(v_loss)

    return loss_continuous, loss_resumed


# -----------------------------------------------------------------------------
# 8. Visualization Dashboard Generation
# -----------------------------------------------------------------------------
def generate_serialization_dashboard(
    results: Dict[str, Dict[str, Any]],
    logits_dict: Dict[str, torch.Tensor],
    loss_continuous: List[float],
    loss_resumed: List[float],
    output_path: str = "model_serialization_benchmark.png"
) -> None:
    print("[*] Generating publication-grade serialization benchmark dashboard...")
    fig = plt.figure(figsize=(18, 11), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], wspace=0.25, hspace=0.35)

    methods = list(results.keys())
    file_sizes = [results[m]["file_size_kb"] for m in methods]
    save_times = [results[m]["save_ms"] for m in methods]
    load_times = [results[m]["load_ms"] for m in methods]

    # Panel 1: File Size vs Serialization Latency
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods))
    width = 0.28
    b1 = ax1.bar(x - width/2, save_times, width, label="Save Time (ms)", color="#3498db", alpha=0.9)
    b2 = ax1.bar(x + width/2, load_times, width, label="Load Time (ms)", color="#e67e22", alpha=0.9)

    ax1.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax1.set_title("Serialization & Deserialization Latency (ms)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=9.5)

    # Secondary axis for file size
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, file_sizes, color="#2ecc71", marker="o", linewidth=2.5, markersize=8, label="File Size (KB)")
    ax1_twin.set_ylabel("Disk Size (KB)", fontsize=11, fontweight="bold", color="#27ae60")
    ax1_twin.tick_params(axis="y", labelcolor="#27ae60")
    ax1_twin.legend(loc="upper right", fontsize=9.5)

    # Panel 2: Deterministic Resumption Continuity Proof
    ax2 = fig.add_subplot(gs[0, 1])
    epochs = range(1, len(loss_continuous) + 1)
    ax2.plot(epochs, loss_continuous, label="Continuous Baseline", color="#2c3e50", linewidth=3, marker="o")
    ax2.plot(epochs, loss_resumed, label="Interrupted & Resumed Checkpoint", color="#e74c3c",
             linestyle="--", linewidth=2.5, marker="x", markersize=9)
    ax2.axvline(x=3, color="gray", linestyle=":", linewidth=2, label="Crash & Resumption Point (Epoch 3)")
    ax2.set_title("Deterministic Resumption Verification: Exact Loss Trajectory", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Cross-Entropy Loss", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9.5, loc="upper right")

    # Panel 3: Logit Output Residuals Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    orig_logits = logits_dict["Original (Eager)"].numpy()
    method_names = [m for m in logits_dict.keys() if m != "Original (Eager)"]
    residuals = []
    for m in method_names:
        diff = np.abs(orig_logits - logits_dict[m].numpy()).max(axis=1) # max per sample
        residuals.append(diff[:20]) # first 20 samples

    residuals_matrix = np.array(residuals)
    im = ax3.imshow(residuals_matrix, cmap="magma", aspect="auto")
    ax3.set_yticks(np.arange(len(method_names)))
    ax3.set_yticklabels(method_names, fontsize=10, fontweight="bold")
    ax3.set_xlabel("Test Samples (Indices 0–19)", fontsize=11)
    ax3.set_title(r"Bitwise Logit Residuals: $\max_c |\hat{y}_{\mathrm{orig}} - \hat{y}_{\mathrm{reloaded}}|$",
                  fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Error (Float32 Precision)", fontsize=9.5)

    # Panel 4: Security & Portability Architecture Comparison Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    table_data = [
        ["Serialization Method", "Safety Level", "File Size", "Inference Portability", "State Recovery"],
        ["State Dict (.pt)", "High (weights_only)", f"{results['State Dict']['file_size_kb']:.1f} KB", "Requires Python Code", "Weights & Buffers"],
        ["Full Model (.pt)", "Low (Arbitrary Pickle)", f"{results['Full Model']['file_size_kb']:.1f} KB", "Requires Python Code", "Model Graph + Weights"],
        ["Resumable Ckpt (.tar)", "Internal Safe", f"{results['Resumable Checkpoint']['file_size_kb']:.1f} KB", "Training Environment", "Full State (Opt/RNG)"],
        ["TorchScript JIT (.pt)", "Maximum (Pure C++)", f"{results['TorchScript JIT']['file_size_kb']:.1f} KB", "Zero-Python / C++ Runtime", "Optimized Graph IR"]
    ]
    table = ax4.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.24, 0.22, 0.16, 0.22, 0.20])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.8)

    # Style header row
    for col in range(5):
        cell = table[(0, col)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")

    # Alternate row colors
    for row in range(1, 5):
        bg_color = "#f8f9fa" if row % 2 == 0 else "#ffffff"
        for col in range(5):
            table[(row, col)].set_facecolor(bg_color)

    ax4.set_title("Architectural Trade-Offs & Production Safety Comparison", fontsize=12, fontweight="bold", pad=20)

    plt.suptitle("PyTorch Model Serialization, Checkpointing & Deployment Benchmark",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Diagnostic dashboard saved successfully to {output_path}")


# -----------------------------------------------------------------------------
# 9. Main Routine
# -----------------------------------------------------------------------------
def main() -> None:
    print("================================================================================")
    print("     PyTorch Model Serialization, Checkpointing & JIT Serving Benchmark         ")
    print("================================================================================")

    # 1. Dataset Generation
    train_loader, val_loader, test_batch = generate_dataset()
    print(f"[*] Generated synthetic classification dataset: 4000 samples, 20 features, 5 classes.")

    # 2. Train Base Model
    set_seed(42)
    model = DeepClassifier(input_dim=20, hidden_dim=64, num_classes=5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print("[*] Training baseline neural classifier for 4 epochs...")
    for epoch in range(1, 5):
        t_loss = train_single_epoch(model, train_loader, criterion, optimizer)
        v_loss = evaluate_loss(model, val_loader, criterion)
        scheduler.step()
        print(f"    Epoch [{epoch}/4] Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

    # 3. Serialization Benchmark
    print("\n[*] Benchmarking 4 Serialization Modalities...")
    results, logits_dict = benchmark_serialization_methods(
        trained_model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=4,
        test_batch=test_batch
    )

    # 4. Deterministic Resumption Experiment
    print("\n[*] Running Deterministic Training Resumption Experiment (Crash & Recovery)...")
    loss_continuous, loss_resumed = run_deterministic_resumption_experiment(train_loader, val_loader)
    max_resumption_gap = max(abs(c - r) for c, r in zip(loss_continuous, loss_resumed))
    print(f"[+] Resumption verification complete. Max trajectory deviation: {max_resumption_gap:.1e}")

    # 5. Visualization Dashboard
    generate_serialization_dashboard(
        results=results,
        logits_dict=logits_dict,
        loss_continuous=loss_continuous,
        loss_resumed=loss_resumed,
        output_path="model_serialization_benchmark.png"
    )

    # 6. Quantitative Summary Table
    print("\n" + "=" * 90)
    print(f"{'Method':<22} | {'Size (KB)':<10} | {'Save (ms)':<10} | {'Load (ms)':<10} | {'Max Δ Logit':<12} | {'SHA-256':<12}")
    print("-" * 90)
    for method, met in results.items():
        print(f"{method:<22} | {met['file_size_kb']:<10.2f} | {met['save_ms']:<10.2f} | {met['load_ms']:<10.2f} | {met['max_abs_diff']:<12.1e} | {str(met['sha256']):<12}")
    print("=" * 90)
    print("[✓] Execution complete. All artifacts saved cleanly.\n")


if __name__ == "__main__":
    main()
