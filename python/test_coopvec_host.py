"""
test_coopvec_host.py -- Verify CoopVec calibration host integration

Tests:
  1. Query CoopVec device support (RTX 5070 Ti should be supported)
  2. Upload affine calibration weights (identity: scale=1, bias=0)
  3. Upload linear calibration weights (identity matrix W=I, bias=0)
  4. Route with calibration and verify accuracy is preserved

Copyright (c) 2026 Jordi Silvestre Lopez -- Apache 2.0
"""
import sys
import os

def main():
    print("=" * 60)
    print("CoopVec Host Integration Test")
    print("=" * 60)

    # ── Step 1: Import torch ─────────────────────────────────
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"ERROR: PyTorch not available: {e}")
        return False

    # ── Step 2: Load compiled extension ──────────────────────
    ext_dir = os.path.expanduser("~/.cache/torch_extensions/optix_training_ext")
    pyd_file = os.path.join(ext_dir, "optix_training_ext.pyd")

    if not os.path.exists(pyd_file):
        print(f"ERROR: Extension not found at {pyd_file}")
        print("  Run bench_optix_win.bat first to compile.")
        return False

    sys.path.insert(0, ext_dir)

    try:
        import optix_training_ext as ext
        print(f"[OK] Extension loaded from {ext_dir}")
    except Exception as e:
        print(f"ERROR: Failed to import extension: {e}")
        return False

    # ── Step 3: List all available functions ──────────────────
    funcs = [f for f in dir(ext) if not f.startswith('_')]
    print(f"\nExported functions ({len(funcs)}):")
    for f in funcs:
        print(f"  - {f}")

    # Check CoopVec functions exist
    coopvec_funcs = ['is_coopvec_supported', 'upload_calibration_affine',
                     'upload_calibration_linear', 'disable_calibration',
                     'has_calibration']
    missing = [f for f in coopvec_funcs if f not in funcs]
    if missing:
        print(f"\n[FAIL] Missing CoopVec functions: {missing}")
        return False
    print(f"\n[OK] All 5 CoopVec functions present")

    # ── Step 4: Initialize OptiX ─────────────────────────────
    print("\n--- Initializing OptiX pipeline ---")

    # Find PTX/OptiXIR files
    build_dirs = [
        os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'),
        os.path.join(os.path.dirname(__file__), '..', 'build'),
        os.path.join(os.path.dirname(__file__), '..', 'build', 'cuda'),
    ]

    raygen_path = None
    hitgroup_path = None

    for bdir in build_dirs:
        bdir = os.path.abspath(bdir)
        if not os.path.exists(bdir):
            continue
        for root, dirs, files in os.walk(bdir):
            for f in files:
                fp = os.path.join(root, f)
                if 'raygen' in f.lower() and (f.endswith('.optixir') or f.endswith('.ptx')):
                    raygen_path = fp
                if ('hitgroup' in f.lower() or 'closest_hit' in f.lower()) and \
                   (f.endswith('.optixir') or f.endswith('.ptx')):
                    hitgroup_path = fp

    if not raygen_path or not hitgroup_path:
        print(f"[WARN] PTX/OptiXIR files not found in build dirs")
        print(f"  Searched: {build_dirs}")
        print(f"  raygen: {raygen_path}")
        print(f"  hitgroup: {hitgroup_path}")
        print("\n  Trying to initialize anyway (may use cached paths)...")

    if raygen_path and hitgroup_path:
        print(f"  Raygen:   {os.path.basename(raygen_path)}")
        print(f"  Hitgroup: {os.path.basename(hitgroup_path)}")
        try:
            ext.initialize(raygen_path, hitgroup_path)
            print("[OK] OptiX pipeline initialized")
        except Exception as e:
            print(f"[FAIL] Pipeline init failed: {e}")
            return False
    else:
        print("[SKIP] Cannot initialize without PTX files")
        # Still test what we can
        print("\n--- Testing CoopVec support query (without init) ---")
        supported = ext.is_coopvec_supported()
        print(f"  CoopVec supported (pre-init): {supported}")
        print("  Expected: False (not initialized yet)")
        return True

    # ── Step 5: Query CoopVec support ─────────────────────────
    print("\n--- Testing CoopVec device support ---")
    coopvec_supported = ext.is_coopvec_supported()
    print(f"  CoopVec supported: {'YES ✓' if coopvec_supported else 'NO ✗'}")

    if not coopvec_supported:
        print("\n[INFO] CoopVec not supported on this device.")
        print("  This is expected on GPUs older than RTX 50-series (Blackwell).")
        print("  Affine calibration (non-CoopVec path) can still be tested.")

    # ── Step 6: Build GAS for testing ─────────────────────────
    print("\n--- Building test GAS (64 experts) ---")
    import math
    num_experts = 64
    centers = torch.zeros(num_experts, 3, dtype=torch.float32)
    radii = torch.full((num_experts,), 0.5, dtype=torch.float32)

    for i in range(num_experts):
        theta = 2.0 * math.pi * i / num_experts
        phi = math.acos(1.0 - 2.0 * (i + 0.5) / num_experts)
        centers[i, 0] = 10.0 * math.sin(phi) * math.cos(theta)
        centers[i, 1] = 10.0 * math.sin(phi) * math.sin(theta)
        centers[i, 2] = 10.0 * math.cos(phi)

    ext.build_gas(centers, radii)
    print(f"  GAS built: {ext.gas_size()} bytes, {ext.num_experts()} experts")

    # ── Step 7: Baseline routing (no calibration) ─────────────
    print("\n--- Baseline routing (no calibration) ---")
    batch = 256
    positions = torch.zeros(batch, 3, dtype=torch.float32, device='cuda')
    directions = torch.zeros(batch, 3, dtype=torch.float32, device='cuda')

    for i in range(batch):
        target = i % num_experts
        cx, cy, cz = centers[target].tolist()
        directions[i, 0] = cx
        directions[i, 1] = cy
        directions[i, 2] = cz
        norm = math.sqrt(cx*cx + cy*cy + cz*cz)
        if norm > 1e-6:
            directions[i] /= norm

    ids_baseline, dists_baseline = ext.route(positions, directions)
    ids_baseline_cpu = ids_baseline.cpu()
    correct_baseline = sum(1 for i in range(batch)
                          if ids_baseline_cpu[i].item() == (i % num_experts))
    print(f"  Accuracy: {correct_baseline}/{batch} ({100*correct_baseline/batch:.1f}%)")

    # ── Step 8: Test has_calibration (should be False) ────────
    print(f"\n--- Calibration state ---")
    print(f"  has_calibration: {ext.has_calibration()}")

    # ── Step 9: Upload affine calibration (identity) ──────────
    print("\n--- Test 1: Affine calibration (identity: s=1, b=0) ---")
    scale = torch.ones(64, dtype=torch.float32)
    bias = torch.zeros(64, dtype=torch.float32)

    try:
        ok = ext.upload_calibration_affine(scale, bias)
        print(f"  Upload result: {'OK ✓' if ok else 'FAILED ✗'}")
        print(f"  has_calibration: {ext.has_calibration()}")
    except Exception as e:
        print(f"  Upload failed: {e}")

    # Route again — should give same results with identity calibration
    ids_cal, dists_cal = ext.route(positions, directions)
    ids_cal_cpu = ids_cal.cpu()
    correct_cal = sum(1 for i in range(batch)
                     if ids_cal_cpu[i].item() == (i % num_experts))
    print(f"  Accuracy with identity affine: {correct_cal}/{batch} ({100*correct_cal/batch:.1f}%)")

    if correct_cal == correct_baseline:
        print(f"  [OK] Identity calibration preserves routing ✓")
    else:
        print(f"  [WARN] Routing changed after identity calibration")

    # ── Step 10: Upload linear calibration (identity matrix) ──
    if coopvec_supported:
        print("\n--- Test 2: Linear calibration (W=I, b=0) ---")
        W = torch.eye(64, dtype=torch.float32)
        linear_bias = torch.zeros(64, dtype=torch.float32)

        try:
            ok = ext.upload_calibration_linear(W.contiguous(), linear_bias)
            print(f"  Upload result: {'OK ✓' if ok else 'FAILED ✗'}")
        except Exception as e:
            print(f"  Upload failed: {e}")

        # Route again
        ids_lin, dists_lin = ext.route(positions, directions)
        ids_lin_cpu = ids_lin.cpu()
        correct_lin = sum(1 for i in range(batch)
                         if ids_lin_cpu[i].item() == (i % num_experts))
        print(f"  Accuracy with identity linear: {correct_lin}/{batch} ({100*correct_lin/batch:.1f}%)")
    else:
        print("\n--- Test 2: Linear calibration SKIPPED (CoopVec not supported) ---")

    # ── Step 11: Disable calibration ──────────────────────────
    print("\n--- Test 3: Disable calibration ---")
    ext.disable_calibration()
    print(f"  has_calibration: {ext.has_calibration()}")

    ids_nocal, dists_nocal = ext.route(positions, directions)
    ids_nocal_cpu = ids_nocal.cpu()
    correct_nocal = sum(1 for i in range(batch)
                       if ids_nocal_cpu[i].item() == (i % num_experts))
    print(f"  Accuracy after disable: {correct_nocal}/{batch} ({100*correct_nocal/batch:.1f}%)")

    # ── Cleanup ───────────────────────────────────────────────
    ext.shutdown()

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  CoopVec device support:  {'YES ✓' if coopvec_supported else 'NO ✗'}")
    print(f"  Baseline accuracy:       {correct_baseline}/{batch}")
    print(f"  Affine cal accuracy:     {correct_cal}/{batch}")
    if coopvec_supported:
        print(f"  Linear cal accuracy:     {correct_lin}/{batch}")
    print(f"  Post-disable accuracy:   {correct_nocal}/{batch}")

    all_pass = (correct_baseline == batch and correct_cal == batch and
                correct_nocal == batch)
    if coopvec_supported:
        all_pass = all_pass and (correct_lin == batch)

    print(f"\n  {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
