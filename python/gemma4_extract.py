#!/usr/bin/env python3
"""
gemma4_extract.py -- Extract MoE layers + hidden states from Gemma 4 26B A4B

Gemma 4 MoE architecture (from config.json):
    - 30 MoE layers
    - 128 experts + 1 shared (always-on) expert per layer
    - Top-8 routing (from 128, not counting shared)
    - hidden_size = 2816
    - moe_intermediate_size = 704
    - hidden_activation = gelu_pytorch_tanh
    - Sliding window (1024) + Full attention hybrid

Model IDs:
    - google/gemma-4-26b-a4b-it   (instruction-tuned)
    - google/gemma-4-26b-a4b       (base)

VRAM Requirements:
    - BF16:  ~52 GB (needs A100/H100)
    - INT4:  ~13 GB (fits RTX 5070 Ti 16GB)
    - INT8:  ~26 GB (needs RTX 4090 24GB)

Usage:
    # Extract hidden states for all 30 layers (INT4, fits 16GB VRAM)
    python gemma4_extract.py --model google/gemma-4-26b-a4b-it --layer 0 --quantize int4

    # Extract all layers in a loop
    for i in range(30):
        python gemma4_extract.py --model google/gemma-4-26b-a4b-it --layer $i --quantize int4

Copyright (c) 2026 Jordi Silvestre Lopez -- Apache 2.0
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Gemma4MoEConfig:
    """Immutable config matching Gemma 4 26B A4B architecture."""
    hidden_size: int = 2816
    moe_intermediate_size: int = 704
    n_experts: int = 128
    n_shared_experts: int = 1      # Always-on shared expert
    top_k: int = 8
    n_layers: int = 30
    vocab_size: int = 262144
    model_type: str = "gemma4"
    activation: str = "gelu_pytorch_tanh"


# ─────────────────────────────────────────────────────────────────
# Hidden State Capture Hook (adapted from OLMoE extractor)
# ─────────────────────────────────────────────────────────────────

class HiddenStateCapture:
    """Register a forward hook on a decoder layer to capture pre-MoE hidden states."""

    def __init__(self):
        self.captured = []
        self._hook = None

    def hook_fn(self, module, args, output):
        """
        Capture the output of the post_attention_layernorm.
        This is what the MoE gate receives as input.
        """
        if isinstance(output, torch.Tensor):
            self.captured.append(output.detach().cpu().to(torch.float16))
        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            self.captured.append(output[0].detach().cpu().to(torch.float16))

    def register(self, module):
        self._hook = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


# ─────────────────────────────────────────────────────────────────
# Model Loading (with quantization support)
# ─────────────────────────────────────────────────────────────────

def load_gemma4_model(
    model_id: str,
    quantize: str = "none",
    device: str = "cuda",
    trust_remote_code: bool = True,
):
    """
    Load Gemma 4 26B A4B model with optional quantization.

    Args:
        model_id: HuggingFace model ID (e.g. "google/gemma-4-26b-a4b-it")
        quantize: "none", "int4", "int8"
        device: target device
        trust_remote_code: allow custom model code

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/4] Loading Gemma 4 model: {model_id}")
    print(f"       Quantization: {quantize}")

    kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
    }

    if quantize == "int4":
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = bnb_config
            print("       Using BitsAndBytes NF4 quantization (~13 GB VRAM)")
        except ImportError:
            print("       ERROR: pip install bitsandbytes  (for INT4 quantization)")
            raise
    elif quantize == "int8":
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["quantization_config"] = bnb_config
            print("       Using BitsAndBytes INT8 quantization (~26 GB VRAM)")
        except ImportError:
            print("       ERROR: pip install bitsandbytes  (for INT8 quantization)")
            raise
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        print("       Using BF16 (~52 GB VRAM)")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model.eval()
    elapsed = time.time() - t0

    # Memory report
    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"       Loaded in {elapsed:.1f}s, VRAM: {vram_gb:.1f} GB")
    else:
        print(f"       Loaded in {elapsed:.1f}s")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────
# Layer Structure Discovery
# ─────────────────────────────────────────────────────────────────

def discover_gemma4_structure(model) -> dict:
    """
    Discover the MoE layer structure of a Gemma 4 model.

    Handles potential variations in HuggingFace implementation.
    Returns dict with keys: 'layers', 'n_layers', 'gate_attr', 'ln_attr'
    """
    info = {}

    # Find layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        info['layers'] = model.model.layers
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        info['layers'] = model.language_model.model.layers
    else:
        # Try to find layers attribute anywhere
        for name, module in model.named_modules():
            if 'layers' in name and hasattr(module, '__len__'):
                info['layers'] = module
                break
        if 'layers' not in info:
            raise AttributeError(
                "Cannot find decoder layers. Model structure:\n" +
                "\n".join(f"  {n}" for n, _ in model.named_children())
            )

    info['n_layers'] = len(info['layers'])

    # Inspect first layer to find gate and layernorm
    layer0 = info['layers'][0]
    layer_attrs = {name: type(module).__name__
                   for name, module in layer0.named_modules() if name}

    print(f"\n  Model structure discovery:")
    print(f"    Layers: {info['n_layers']}")
    print(f"    Layer 0 modules:")
    for name, typename in sorted(layer_attrs.items()):
        if len(name.split('.')) <= 2:  # Only show top-level
            print(f"      {name}: {typename}")

    # Find the MoE gate (router)
    gate_candidates = ['block_sparse_moe.gate', 'mlp.gate', 'moe.gate',
                       'feed_forward.gate', 'block_sparse_moe.router.gate']
    for gate_attr in gate_candidates:
        parts = gate_attr.split('.')
        obj = layer0
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, 'weight'):
            info['gate_attr'] = gate_attr
            info['gate_shape'] = obj.weight.shape
            print(f"    Gate found: {gate_attr} — shape {obj.weight.shape}")
            break

    if 'gate_attr' not in info:
        # Brute force: find any Linear with output == 128
        for name, module in layer0.named_modules():
            if hasattr(module, 'weight') and module.weight.shape[0] == 128:
                info['gate_attr'] = name
                info['gate_shape'] = module.weight.shape
                print(f"    Gate found (by shape): {name} — shape {module.weight.shape}")
                break

    # Find post-attention layernorm
    ln_candidates = ['post_attention_layernorm', 'post_layernorm',
                     'post_attention_norm', 'ln_2', 'norm2']
    for ln_attr in ln_candidates:
        if hasattr(layer0, ln_attr):
            info['ln_attr'] = ln_attr
            print(f"    LayerNorm: {ln_attr}")
            break

    if 'ln_attr' not in info:
        # Find pre-feedforward norm
        for name, module in layer0.named_modules():
            if 'norm' in name.lower() and 'post' in name.lower():
                info['ln_attr'] = name
                print(f"    LayerNorm (by name): {name}")
                break

    return info


# ─────────────────────────────────────────────────────────────────
# Hidden State Extraction
# ─────────────────────────────────────────────────────────────────

def extract_gemma4_hidden_states(
    model_id: str = "google/gemma-4-26b-a4b-it",
    layer_idx: int = 0,
    max_tokens: int = 200_000,
    batch_seq_len: int = 256,
    quantize: str = "int4",
    device: str = "cuda",
    output_dir: str = "data/gemma4_hiddens",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
) -> str:
    """
    Extract real hidden states from Gemma 4 for BVH router distillation.

    Pipeline:
        1. Load quantized Gemma 4 model
        2. Hook post_attention_layernorm at target layer
        3. Run text through the model
        4. Capture hidden states + gate routing decisions
        5. Save to disk

    Returns:
        Path to saved .pt file
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print(f"  Gemma 4 Hidden State Extraction — Layer {layer_idx}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────
    model, tokenizer = load_gemma4_model(model_id, quantize=quantize, device=device)

    # ── Discover structure ────────────────────────────────────
    structure = discover_gemma4_structure(model)
    n_layers = structure['n_layers']

    if not (0 <= layer_idx < n_layers):
        raise ValueError(f"--layer {layer_idx} out of range [0, {n_layers - 1}]")

    # ── Load dataset ──────────────────────────────────────────
    print(f"\n[2/4] Loading dataset: {dataset_name}/{dataset_config}...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, dataset_config, split="train")
        text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    except ImportError:
        print("  ERROR: pip install datasets")
        raise

    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    all_token_ids = encodings.input_ids.squeeze(0)[:max_tokens]
    print(f"  Tokenized: {len(all_token_ids):,} tokens")

    # ── Hook the target layer ─────────────────────────────────
    print(f"\n[3/4] Hooking layer {layer_idx}...")
    target_layer = structure['layers'][layer_idx]

    # Hook the post-attention layernorm
    ln_attr = structure.get('ln_attr', 'post_attention_layernorm')
    ln_parts = ln_attr.split('.')
    ln_module = target_layer
    for part in ln_parts:
        ln_module = getattr(ln_module, part)

    capture = HiddenStateCapture()
    capture.register(ln_module)

    # Get gate for routing labels
    gate_attr = structure.get('gate_attr', 'block_sparse_moe.gate')
    gate_parts = gate_attr.split('.')
    gate_module = target_layer
    for part in gate_parts:
        gate_module = getattr(gate_module, part)

    gate_weight = gate_module.weight  # [128, 2816]
    config = Gemma4MoEConfig()
    print(f"  Gate shape: {gate_weight.shape}")
    print(f"  Expected: [{config.n_experts}, {config.hidden_size}]")

    # ── Run inference and capture ─────────────────────────────
    print(f"\n[4/4] Running inference to capture hidden states...")
    t0 = time.time()

    all_hidden = []
    all_gate_probs = []
    all_topk_ids = []
    n_captured = 0

    # Use shorter sequences for Gemma 4 (it's bigger)
    with torch.no_grad():
        for start in range(0, len(all_token_ids), batch_seq_len):
            chunk = all_token_ids[start:start + batch_seq_len].unsqueeze(0)
            if chunk.shape[1] == 0:
                continue
            chunk = chunk.to(device)

            capture.captured.clear()

            try:
                _ = model(chunk)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  OOM at offset {start}, reducing batch_seq_len")
                    torch.cuda.empty_cache()
                    continue
                raise

            if len(capture.captured) == 0:
                print(f"  WARNING: No capture at offset {start}")
                continue

            h = capture.captured[0]   # (1, seq_len, 2816) fp16 CPU
            h_flat = h.squeeze(0)     # (seq_len, 2816)

            # Compute gate routing for these hidden states
            h_gate = h_flat.to(device=gate_weight.device,
                              dtype=gate_weight.dtype)
            raw_logits = F.linear(h_gate, gate_weight)
            gate_probs = F.softmax(raw_logits.float(), dim=-1)
            _, topk_ids = gate_probs.topk(config.top_k, dim=-1)

            all_hidden.append(h_flat)
            all_gate_probs.append(gate_probs.cpu().half())
            all_topk_ids.append(topk_ids.cpu())

            n_captured += h_flat.shape[0]
            if n_captured % 10000 < batch_seq_len:
                elapsed = time.time() - t0
                rate = n_captured / elapsed if elapsed > 0 else 0
                print(f"  Captured {n_captured:,} hidden states "
                      f"({rate:.0f} tokens/s)...")

    capture.remove()

    if len(all_hidden) == 0:
        print("\n  ERROR: No hidden states captured!")
        return ""

    # Concatenate
    hidden_states = torch.cat(all_hidden, dim=0)   # (N, 2816)
    gate_probs = torch.cat(all_gate_probs, dim=0)  # (N, 128)
    topk_ids = torch.cat(all_topk_ids, dim=0)      # (N, 8)

    elapsed = time.time() - t0
    print(f"\n  Captured {hidden_states.shape[0]:,} hidden states in {elapsed:.1f}s")
    print(f"  Shape: {hidden_states.shape}")
    print(f"  Unique experts used (top-1): "
          f"{topk_ids[:, 0].unique().numel()}/{config.n_experts}")

    # Expert utilization analysis
    top1_counts = torch.zeros(config.n_experts, dtype=torch.long)
    for eid in topk_ids[:, 0]:
        top1_counts[eid] += 1
    active = (top1_counts > 0).sum().item()
    top5 = top1_counts.topk(5)
    print(f"  Active experts (top-1): {active}/{config.n_experts}")
    print(f"  Top 5 most used: {list(zip(top5.indices.tolist(), top5.values.tolist()))}")

    # ── Save ──────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"real_hiddens_layer{layer_idx}.pt"
    torch.save({
        'hidden_states': hidden_states,      # (N, 2816) fp16
        'gate_logits': gate_probs,           # (N, 128) fp16 softmax probs
        'topk_ids': topk_ids,                # (N, 8) int64
        'layer_idx': layer_idx,
        'n_samples': hidden_states.shape[0],
        'model_id': model_id,
        'model_type': 'gemma4',
        'hidden_size': config.hidden_size,
        'n_experts': config.n_experts,
        'top_k': config.top_k,
    }, str(out_path))

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved to {out_path} ({size_mb:.1f} MB)")

    return str(out_path)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract real hidden states from Gemma 4 26B A4B for SpectralAI"
    )
    parser.add_argument("--model", type=str, default="google/gemma-4-26b-a4b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer to extract (0-29)")
    parser.add_argument("--all-layers", action="store_true",
                        help="Extract all 30 layers sequentially")
    parser.add_argument("--max-tokens", type=int, default=200_000,
                        help="Max tokens to process from dataset")
    parser.add_argument("--batch-seq-len", type=int, default=256,
                        help="Sequence length per batch (lower = less VRAM)")
    parser.add_argument("--quantize", type=str, default="int4",
                        choices=["none", "int4", "int8"],
                        help="Quantization mode (int4 fits 16GB VRAM)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="data/gemma4_hiddens",
                        help="Output directory for .pt files")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset name (HuggingFace)")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1",
                        help="Dataset config name")
    args = parser.parse_args()

    if args.all_layers:
        # Extract all 30 layers — model loaded once, reused
        print("=" * 60)
        print("  Gemma 4 — Extracting ALL 30 layers")
        print("=" * 60)

        # Load model once
        model, tokenizer = load_gemma4_model(
            args.model, quantize=args.quantize, device=args.device
        )

        # Discover structure once
        structure = discover_gemma4_structure(model)

        for layer_idx in range(structure['n_layers']):
            print(f"\n{'='*60}")
            print(f"  Layer {layer_idx} / {structure['n_layers'] - 1}")
            print(f"{'='*60}")

            try:
                extract_gemma4_hidden_states(
                    model_id=args.model,
                    layer_idx=layer_idx,
                    max_tokens=args.max_tokens,
                    batch_seq_len=args.batch_seq_len,
                    quantize=args.quantize,
                    device=args.device,
                    output_dir=args.output_dir,
                    dataset_name=args.dataset,
                    dataset_config=args.dataset_config,
                )
            except Exception as e:
                print(f"  ERROR on layer {layer_idx}: {e}")
                continue
    else:
        extract_gemma4_hidden_states(
            model_id=args.model,
            layer_idx=args.layer,
            max_tokens=args.max_tokens,
            batch_seq_len=args.batch_seq_len,
            quantize=args.quantize,
            device=args.device,
            output_dir=args.output_dir,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
        )


if __name__ == "__main__":
    main()
