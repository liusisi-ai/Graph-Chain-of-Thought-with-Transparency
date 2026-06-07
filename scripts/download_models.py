"""Download LLM and sentence-transformer model weights to a local directory.

Usage on AutoDL (where direct connection to huggingface.co fails):

    python scripts/download_models.py

The script:
  1. forces ``HF_ENDPOINT=https://hf-mirror.com``
  2. downloads ``lmsys/vicuna-7b-v1.5-16k`` to /root/autodl-tmp/vicuna-7b-v1.5-16k
  3. downloads ``sentence-transformers/all-mpnet-base-v2``
     to /root/autodl-tmp/all-mpnet-base-v2

Once finished, ``config.py`` will detect the local copies and skip the network
entirely on subsequent ``python main.py`` runs.

Environment overrides:
  HF_ENDPOINT       – mirror to use (default https://hf-mirror.com)
  AUTODL_BASE       – base dir for local copies (default /root/autodl-tmp)
  ONLY              – comma-separated list to limit which models to fetch:
                      ``ONLY=sbert`` or ``ONLY=llm`` or ``ONLY=sbert,llm``
"""

from __future__ import annotations

import os
import sys

# Force the mirror BEFORE huggingface_hub gets imported
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

AUTODL_BASE = os.environ.get("AUTODL_BASE", "/root/autodl-tmp")
ONLY = set(filter(None, os.environ.get("ONLY", "").split(",")))


MODELS = [
    # tag,  hf_id,                                 local_dir_name
    ("sbert", "sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2"),
    ("llm",   "lmsys/vicuna-7b-v1.5-16k",                "vicuna-7b-v1.5-16k"),
]


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is missing.\n"
              "  pip install -U 'huggingface_hub[hf_transfer]'", file=sys.stderr)
        return 1

    print(f"🌐 HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")
    print(f"📁 Local base = {AUTODL_BASE}")

    os.makedirs(AUTODL_BASE, exist_ok=True)

    for tag, repo_id, local_name in MODELS:
        if ONLY and tag not in ONLY:
            print(f"\n⏭  Skipping {tag} (ONLY={','.join(sorted(ONLY))})")
            continue
        local_dir = os.path.join(AUTODL_BASE, local_name)
        if os.path.isdir(local_dir) and any(os.scandir(local_dir)):
            print(f"\n✅ Already present: {local_dir} (skip)")
            continue
        print(f"\n⏬ Downloading {repo_id} → {local_dir}")
        try:
            # Older / newer huggingface_hub APIs differ – try the new signature
            # first (>=1.0 dropped local_dir_use_symlinks/resume_download).
            try:
                snapshot_download(repo_id=repo_id, local_dir=local_dir, max_workers=8)
            except TypeError:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            print(f"✅ Done: {local_dir}")
        except Exception as e:
            print(f"❌ Failed to download {repo_id}: {e}", file=sys.stderr)
            print(
                "   - 检查 HF_ENDPOINT 是否可访问\n"
                "   - 若 hf-mirror 限流，可改用 modelscope:\n"
                "       pip install modelscope\n"
                "       modelscope download --model AI-ModelScope/"
                f"{local_name} --local_dir {local_dir}",
                file=sys.stderr,
            )
            return 2
    print("\n🎉 All requested models are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
