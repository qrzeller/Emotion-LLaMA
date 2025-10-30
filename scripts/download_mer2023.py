#!/usr/bin/env python3
import os
import sys
from huggingface_hub import snapshot_download, login

BASE_DIR = "/export/home/scratch/qze/datasets"
TARGET = os.path.abspath(os.path.join(BASE_DIR, "MER2023"))

def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        print("Using token from environment for login…", flush=True)
        login(token=token, add_to_git_credential=True)
        print("Login succeeded.", flush=True)
    else:
        print("No token provided; attempting download without login (may fail if gated).", flush=True)

    print(f"Downloading MERChallenge/MER2023 to: {TARGET}", flush=True)
    try:
        snapshot_download(
            repo_id="MERChallenge/MER2023",
            repo_type="dataset",
            local_dir=TARGET,
            local_dir_use_symlinks=False,
        )
        print("Download complete.", flush=True)
        return 0
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("If access is gated, set your Hugging Face token and rerun:", file=sys.stderr)
        print("  tcsh: setenv HUGGINGFACE_HUB_TOKEN <your_hf_token>", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
