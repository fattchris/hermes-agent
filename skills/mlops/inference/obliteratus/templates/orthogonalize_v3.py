#!/usr/bin/env python3
"""
orthogonalize_v3.py — layer-list-aware weight orthogonalization for hybrid SSM models.

KEY DIFFERENCE from v2: --layers accepts explicit comma-separated indices
(e.g. 3,7,11,15,19,23,27,31) instead of a range (10-30). This lets you
target ONLY the 16 attention layers while skipping all 48 SSM layers.

Usage:
  python3 orthogonalize_v3.py \
    --src model_f16.gguf \
    --vec refusal_mean.gguf \
    --out model_ablit.gguf \
    --targets attn_output.weight \
    --layers 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63 \
    --alpha 1.0

NEVER include ssm_out.weight in --targets on hybrid SSM models.
ALWAYS check the layer layout first (see references/manual-gguf-orthogonalization.md).
"""
import sys, os, re, argparse, shutil, numpy as np
from gguf import GGUFReader

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source F16 GGUF")
    ap.add_argument("--vec", required=True, help="Refusal direction vectors (cvector-generator output)")
    ap.add_argument("--out", required=True, help="Output GGUF path")
    ap.add_argument("--targets", default="attn_output.weight",
                    help="Comma-separated tensor suffixes (NOT ssm_out!)")
    ap.add_argument("--layers", required=True,
                    help="Comma-separated layer indices, e.g. 3,7,11,15")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Projection strength (0=none, 1=full removal, 0.7=partial)")
    return ap.parse_args()

def main():
    args = parse_args()
    targets = set(args.targets.split(","))
    layer_set = set(int(x) for x in args.layers.split(","))
    alpha = args.alpha
    N_EMBD = 5120  # model-specific — adjust for other models

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
    print(f"Layers ({len(layer_set)}): {sorted(layer_set)}")
    print(f"Alpha: {alpha}")

    # Copy source (hard link for APFS copy-on-write, fallback to copy)
    if os.path.exists(args.out):
        os.unlink(args.out)
    try:
        os.link(args.src, args.out)
        print(f"Hard-linked -> {args.out}")
    except OSError:
        shutil.copy2(args.src, args.out)
        print(f"Copied -> {args.out}")

    # Orthogonalize in place
    r = GGUFReader(args.out, mode="r+")
    modified = 0
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.(.+)$", t.name)
        if not m:
            continue
        layer = int(m.group(1))
        suffix = m.group(2)
        if suffix not in targets:
            continue
        if layer not in layer_set:
            continue

        # direction.N = activation after layer N-1, so for layer L use L+1
        dir_idx = min(layer + 1, max(dirs))
        if dir_idx not in dirs:
            continue
        rhat = dirs[dir_idx]

        D = t.data  # numpy view, shape (n_embd, in)
        assert D.shape[0] == N_EMBD, (t.name, D.shape)
        Df = D.astype(np.float32)
        proj = rhat @ Df              # (in,) = rhat^T D
        Df -= alpha * np.outer(rhat, proj)  # D - alpha * rhat(rhat^T D)
        D[:] = Df.astype(np.float16)  # write back through mmap
        modified += 1
        print(f"  blk.{layer}.{suffix}: dir={dir_idx} |proj|={np.linalg.norm(proj):.2f}")

    del r  # flush mmap
    print(f"\nModified {modified} matrices")
    print(f"Output: {args.out}")

if __name__ == "__main__":
    main()
