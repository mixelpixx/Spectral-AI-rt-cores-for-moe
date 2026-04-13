# SpectralAI — Session Documentation

**Author:** Jordi Silvestre Lopez (jsilvestre@silmisa.com)  
**Date:** 2026-04-13  
**Platform:** Windows 11, RTX 5070 Ti 16GB (Blackwell sm_120)  
**Toolchain:** MSVC 2022, CUDA 13.2, OptiX 9.1, PyTorch 2.11.0+cu128

---

## Table of Contents

1. [OptiX Cooperative Vectors (CoopVec)](#1-optix-cooperative-vectors-coopvec)
2. [BranchSpecificBVHRouter Retraining](#2-branchspecificbvhrouter-retraining)
3. [Gemma 4 26B A4B Migration](#3-gemma-4-26b-a4b-migration)
4. [File Inventory](#4-file-inventory)
5. [Execution Guide](#5-execution-guide)

---

## 1. OptiX Cooperative Vectors (CoopVec)

### Objective

Eliminate the ~1-2ms PyTorch CPU/GPU round-trip for post-routing calibration by performing
logit calibration **in-shader** using Tensor Cores via OptiX Cooperative Vectors.

### What Was Implemented

#### Host-side (`cuda/optix_router_host.cpp`)

Three new methods in `RTCoreRouter`:

| Method | Purpose |
|---|---|
| `isCoopVecSupported()` | Queries `OPTIX_DEVICE_PROPERTY_COOP_VEC` to check Tensor Core CoopVec availability |
| `uploadCalibrationAffine(scale, bias)` | Per-expert FP32→FP16 scale+bias upload (128 params) |
| `uploadCalibrationLinear(W, bias)` | Full 128×128 matrix: FP32→FP16→`optixCoopVecMatrixConvert`→`INFERENCING_OPTIMAL` |

Supporting infrastructure:
- `HostCalibrationWeights` struct — 64-byte aligned, contains mode, affine arrays, linear matrix pointer, bias
- `uploadCalibrationToDevice()` — Internal helper for device memory management  
- `disableCalibration()` — Revert to pass-through mode
- Cleanup integrated into `RTCoreRouter` destructor (frees `d_calibration_weights_`, `d_calibration_matrix_`)

#### CoopVec Pipeline Flow

```
Host (CPU)                          Device (GPU Tensor Cores)
──────────                          ─────────────────────────
W_fp32[NxK]                         
    │                               
__float2half() → W_fp16             
    │                               
cudaMemcpy → d_W_rowmajor           
    │                               
optixCoopVecMatrixComputeSize()     
    → optimal_size (8192 bytes for 64x64)
    │                               
optixCoopVecMatrixConvert()         
    ROW_MAJOR → INFERENCING_OPTIMAL 
    │                               
d_calibration_matrix_ on GPU ──────→ optixCoopVecMatMul()
                                     (in closest_hit shader)
                                         │
                                    calibrated_logits
```

#### Python Bindings (`cuda/v5/optix_training_ext.cu`)

Five new pybind11 functions added:

```python
import optix_training_ext as ext

ext.is_coopvec_supported()                    # → bool
ext.upload_calibration_affine(scale, bias)     # → bool (CPU tensors, float32)  
ext.upload_calibration_linear(W, bias)         # → bool (CPU tensors, float32)
ext.disable_calibration()                      # → bool
ext.has_calibration()                          # → bool
```

#### Build Changes (`cuda/v5/build_optix_ext.py`)

- `#include <cuda_fp16.h>` added to host for `half` type
- Function list print now dynamic (shows all 16 exported functions)
- `SPECTRAL_NUM_EXPERTS` updated from 64 to 128 (for Gemma 4 compatibility)

### Verification Results

Test script: `python/test_coopvec_host.py`

```
CoopVec device support:  YES ✓     (RTX 5070 Ti Blackwell)
Matrix convert:          64x64 FP16 → 8192 bytes INFERENCING_OPTIMAL
Affine upload:           448 bytes, mode=AFFINE ✓
Linear upload:           448 bytes + 8KB matrix, mode=LINEAR ✓

Baseline accuracy:       256/256 (100%)
Affine cal accuracy:     256/256 (100%)  ← identity preserves routing
Linear cal accuracy:     256/256 (100%)  ← identity preserves routing
Post-disable accuracy:   256/256 (100%)

ALL TESTS PASSED ✓
```

### OptiX APIs Used

| API | Purpose |
|---|---|
| `optixCoopVecMatrixComputeSize()` | Calculate INFERENCING_OPTIMAL buffer size |
| `optixCoopVecMatrixConvert()` | Convert ROW_MAJOR FP16 → hardware-optimal layout |
| `optixDeviceContextGetProperty(OPTIX_DEVICE_PROPERTY_COOP_VEC)` | Runtime support query |

---

## 2. BranchSpecificBVHRouter Retraining

### Objective

Replace the single-projection `EnhancedBVHRouter` (3D routing) with `BranchSpecificBVHRouter` 
(21-41 projections, 27D effective routing) to improve expert utilization from ~43% to ~98%.

### Architecture: BranchSpecificBVHRouter

```
Level 1: 1 global projection     (embed_dim → 3D)
Level 2: n_level1 projections    (embed_dim → 3D each, selected by L1 branch)
Level 3: n_level1×n_level2 proj  (embed_dim → 3D each, selected by L2 branch)

Training:  weighted average of all projections (gradient flows to all branches)
Inference: hard branch selection (only active projection computed)
```

| Config | OLMoE (64 experts) | Gemma 4 (128 experts) |
|---|---|---|
| Tree | 4×4×4 | 8×4×4 |
| Projections | 1 + 4 + 16 = **21** | 1 + 8 + 32 = **41** |
| Parameters | **4,462,419** | **8,471,139** |
| Effective dims | 3³ = **27D** | 3³ = **27D** |

### Code Changes

#### `python/olmoe_bvh_distill.py`

New CLI arguments:

| Flag | Default | Description |
|---|---|---|
| `--branch-specific` | `false` | Use BranchSpecificBVHRouter instead of Enhanced |
| `--n-experts` | `64` | Number of experts (64=OLMoE, 128=Gemma4) |
| `--embed-dim` | `2048` | Hidden dimension (2048=OLMoE, 2816=Gemma4) |

Auto-factorization of BVH tree from `--n-experts`:
- 64 → 4×4×4
- 128 → 8×4×4  
- 256 → 4×8×8

The training function (`train_bvh_distillation`) is **router-agnostic** — it calls 
`router(hidden_states)` and compares output to gate logits. No changes needed there.

### Pre-existing Data (Ready to Use)

```
data/real_hiddens_layer{0..15}.pt   — 16 files × ~818 MB each (OLMoE)
```

### Execution

```batch
:: Double-click or run from terminal:
train_all_branch_specific.bat

:: Or single layer:
python python/olmoe_bvh_distill.py ^
    --layer 8 ^
    --real-data data/real_hiddens_layer8.pt ^
    --no-upcycle --spectral --branch-specific ^
    --epochs 30 --batch-size 2048 ^
    --save-dir checkpoints/olmoe_distill_branch
```

**Estimated time:** ~80 min for all 16 layers (RTX 5070 Ti)  
**Output:** `checkpoints/olmoe_distill_branch/bvh_router_L{0..15}_best.pt`

---

## 3. Gemma 4 26B A4B Migration

### Why Gemma 4

- Released **April 2, 2026** (vs OLMoE Aug 2024)
- **128 experts** doubles routing complexity → SpectralAI's O(log N) more valuable
- Google DeepMind credibility (vs academic AI2)
- Apache 2.0 license
- 128 experts + 1 shared (always-on) expert

### Architecture Details (from `config.json`)

```json
{
  "hidden_size": 2816,
  "num_experts": 128,
  "top_k_experts": 8,
  "num_hidden_layers": 30,
  "moe_intermediate_size": 704,
  "hidden_activation": "gelu_pytorch_tanh",
  "sliding_window": 1024,
  "max_position_embeddings": 262144
}
```

HuggingFace model ID: `google/gemma-4-26b-a4b-it`

### VRAM for Hidden State Extraction

The full model must be loaded once to extract the hidden states that the
attention layers produce before the MoE block. The attention layers are
dense (shared across all tokens), so they must be loaded. After extraction,
the model is never needed again — the BVH router training uses only ~2 GB.

| Loading mode | VRAM | Speed | Notes |
|---|---|---|---|
| BF16 + CPU offload (`--quantize none`) | GPU + system RAM | ~3x slower | Full precision |
| INT4 bitsandbytes (`--quantize int4`) | ~13 GB (fits 16GB) | Normal | Default. Negligible loss for routing labels |
| INT8 (`--quantize int8`) | ~26 GB | Normal | Minimal loss |

For our use case (extracting gate routing decisions "which 8 of 128 experts?"),
INT4 vs BF16 gives identical results — it's a ranking, not text generation.

### New Files

#### `python/gemma4_extract.py`

Features:
- Auto-discovers model structure (handles HF implementation variations)
- INT4/INT8/BF16 quantization via `--quantize` flag
- `--all-layers` extracts all 30 layers in one run (model loaded once)
- Expert utilization analysis in output
- Saves metadata (model_id, hidden_size, n_experts) in `.pt`

Usage:
```bash
# All 30 layers, INT4 (fits 16GB VRAM)
python python/gemma4_extract.py --all-layers --quantize int4

# Full precision with CPU offloading (no quantization needed)
python python/gemma4_extract.py --all-layers --quantize none

# Single layer
python python/gemma4_extract.py --layer 15 --quantize int4
```

Output: `data/gemma4_hiddens/real_hiddens_layer{0..29}.pt`

#### `train_all_gemma4.bat`

Trains 30 BranchSpecificBVHRouter layers for Gemma 4 (8×4×4 = 128 experts).

Requires: Pre-extracted hidden states from `gemma4_extract.py`  
**Estimated time:** ~150 min for 30 layers

### Prerequisites

```bash
pip install bitsandbytes    # For INT4 quantization (optional if using --quantize none)
pip install datasets        # For WikiText-2 data
# transformers and torch already installed
```

---

## 4. File Inventory

### Files Created This Session

| File | Lines | Purpose |
|---|---|---|
| `python/gemma4_extract.py` | ~350 | Gemma 4 hidden state extractor |
| `python/test_coopvec_host.py` | ~250 | CoopVec verification test |
| `train_all_branch_specific.bat` | ~70 | OLMoE 16-layer training batch |
| `train_all_gemma4.bat` | ~70 | Gemma 4 30-layer training batch |

### Files Modified This Session

| File | Changes |
|---|---|
| `cuda/optix_router_host.cpp` | +230 lines: CoopVec calibration methods, HostCalibrationWeights struct |
| `cuda/v5/optix_training_ext.cu` | +90 lines: 5 CoopVec pybind11 wrappers |
| `cuda/v5/build_optix_ext.py` | Dynamic function list print |
| `python/olmoe_bvh_distill.py` | +45 lines: `--branch-specific`, `--n-experts`, `--embed-dim` flags |

---

## 5. Execution Guide

### Step 1: OLMoE BranchSpecific Retraining (when GPU is free)

```batch
cd "j:\Proyectos\SPECTRAL AI"
train_all_branch_specific.bat
:: ~80 min, outputs to checkpoints/olmoe_distill_branch/
```

### Step 2: Install Gemma 4 Dependencies

```bash
pip install bitsandbytes datasets
```

### Step 3: Extract Gemma 4 Hidden States

```bash
python python/gemma4_extract.py --all-layers --quantize int4
:: ~2-3 hours, outputs to data/gemma4_hiddens/
```

### Step 4: Train Gemma 4 BVH Routers

```batch
train_all_gemma4.bat
:: ~150 min, outputs to checkpoints/gemma4_distill_branch/
```

### Step 5: E2E Evaluation (both models)

### Step 6: Zenodo v2 Publication

Update the three preprints with dual-model results:
- 10.5281/zenodo.19457288 — Main SpectralAI paper
- 10.5281/zenodo.19457411 — Spectral routing
- 10.5281/zenodo.19457473 — Expert specialization

---

## RT Core Benchmark Results (Latest)

| Mode | Latency | Accuracy | Throughput |
|---|---|---|---|
| AABB sync | 36.7 µs | 100% | 6.98 Mq/s |
| AABB async | 20.5 µs | 100% | 12.47 Mq/s |
| Triangle sync | 30.5 µs | 100% | 8.41 Mq/s |
| **Triangle async** | **21.0 µs** | **100%** | **12.20 Mq/s** |

Extension: 16 functions exported (including 5 CoopVec)
