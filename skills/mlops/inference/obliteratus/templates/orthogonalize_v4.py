#!/usr/bin/env python3
"""
orthogonalize_v4.py — layer-aware orthogonalization with fixes:
1. Uses shutil.copy2 (NEVER hard-link — see SKILL.md pitfall)
2. Handles non-blk tensors (output.weight, token_embd.weight)
3. Supports multiple targets including non-layer tensors
4. NaN guards on projection

Usage:
  # Target specific layers
  python3 orthogonalize_v4.py --targets ssm_out.weight --layers 0,1,2,4,5,6 --alpha 1.0 --out out.gguf
  # Target non-blk tensor (LM head)
  python3 orthogonalize_v4.py --targets output.weight --alpha 1.0 --out out.gguf
  # Combo: all three target types
  python3 orthogonalize_v4.py --targets ssm_out.weight,ffn_down.weight,output.weight --alpha 1.0 --out out.gguf
"""
import sys, os, re, argparse, shutil, numpy as np
from gguf import GGUFReader

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source GGUF (must be clean/unmodified)")
    ap.add_argument("--vec", required=True, help="Refusal directions GGUF")
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="attn_output.weight",
        help="Comma-separated tensor suffixes (e.g. ssm_out.weight,output.weight)")
    ap.add_argument("--layers", default=None,
        help="Comma-separated layer indices. If omitted, targets ALL layers for blk tensors.")
    ap.add_argument("--alpha", type=float, default=1.0)
    return ap.parse_args()

def main():
    args = parse_args()
    targets = set(args.targets.split(","))
    alpha = args.alpha
    N_EMBD = 5120  # Override per model if different

    # Load refusal directions
    vr = GGUFReader(args.vec)
    dirs = {}
    for t in vr.tensors:
        m = re.match(r"direction\.(\d+)$", t.name)
        if m:
            v = np.array(t.data, dtype=np.float32).reshape(-1)
            dirs[int(m.group(1))] = v / (np.linalg.norm(v) + 1e-8)
    print(f"Loaded {len(dirs)} refusal directions (idx {min(dirs)}..{max(dirs)})")
    print(f"Targets: {targets}")
    if args.layers:
        layer_set = set(int(x) for x in args.layers.split(","))
        print(f"Layers ({len(layer_set)}): {sorted(layer_set)[:10]}...")
    else:
        layer_set = None
        print("Layers: ALL (no filter)")
    print(f"Alpha: {alpha}")

    # Copy source (NEVER hard-link — os.link corrupts source via shared inode)
    if os.path.exists(args.out):
        os.unlink(args.out)
    shutil.copy2(args.src, args.out)
    print(f"Copied -> {args.out}")

    # Verify source and output don't share inode
    assert os.stat(args.src).st_ino != os.stat(args.out).st_ino, \
        "FATAL: Source and output share inode — hard link bug!"

    r = GGUFReader(args.out, mode="r+")
    modified = 0
    for t in r.tensors:
        # Try blk.N.suffix pattern
        m = re.match(r"blk\.(\d+)\.(.+)$", t.name)
        if m:
            layer = int(m.group(1))
            suffix = m.group(2)
            if suffix not in targets:
                continue
            if layer_set is not None and layer not in layer_set:
                continue
            dir_idx = min(layer + 1, max(dirs))
            if dir_idx not in dirs:
                continue
        else:
            # Non-blk tensor (output.weight, token_embd.weight, etc.)
            if t.name not in targets:
                continue
            # Use last layer's direction for output.weight
            dir_idx = max(dirs)
            layer = -1
            suffix = t.name

        rhat = dirs[dir_idx]
        D = t.data
        Df = D.astype(np.float32)

        # Handle shape: output.weight is (vocab, N_EMBD), need to project along N_EMBD
        if Df.shape[0] == N_EMBD:
            # Standard: (N_EMBD, ...) — project rows
            proj = rhat @ Df
        elif Df.shape[1] == N_EMBD:
            # Transposed: (..., N_EMBD) — project columns
            proj = Df @ rhat
        else:
            print(f"  SKIP {t.name}: shape {Df.shape} doesn't match N_EMBD={N_EMBD}")
            continue

        # NaN guard — prevents zero-projection bug at alpha < 1.0
        if np.any(np.isnan(proj)) or np.any(np.isinf(proj)):
            print(f"  WARNING: numerical instability at {t.name}, skipping")
            continue

        if Df.shape[0] == N_EMBD:
            Df -= alpha * np.outer(rhat, proj)
        else:
            Df -= alpha * np.outer(proj, rhat)

        D[:] = Df.astype(np.float16)
        modified += 1
        if m:
            print(f"  blk.{layer}.{suffix}: dir={dir_idx} |proj|={np.linalg.norm(proj):.2f}")
        else:
            print(f"  {t.name}: dir={dir_idx} |proj|={np.linalg.norm(proj):.2f}")

    del r
    print(f"\nModified {modified} matrices")
    print(f"Output: {args.out}")

if __name__ == "__main__":
    main()
