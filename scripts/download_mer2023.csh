#!/usr/bin/env tcsh
# Download MER2023 dataset via Hugging Face using huggingface_hub snapshot_download.
# Usage (tcsh):
#   # optionally set a token if needed for access
#   # setenv HUGGINGFACE_HUB_TOKEN <your_hf_token>
#   scripts/download_mer2023.csh

# Resolve repo root
set real0 = `which $0`
set scriptdir = `dirname "$real0"`
cd "$scriptdir/.." || exit 1
set repo = "$cwd"

# Ensure venv
if (! -e "$repo/.venv/bin/activate.csh") then
  echo "Error: .venv not found at $repo/.venv. Create it and install requirements first."
  echo "For example (tcsh):"
  echo "  python3 -m venv .venv && source .venv/bin/activate.csh && rehash && pip install -r requirements.txt"
  exit 1
endif
source "$repo/.venv/bin/activate.csh"
rehash

# Install huggingface_hub
pip install -q huggingface_hub || exit 1

# Target download dir
set base_dir = "/export/home/scratch/qze/datasets"
set target = "$base_dir/MER2023"
mkdir -p "$base_dir" || exit 1

# Optional non-interactive login if token provided
if ( $?HUGGINGFACE_HUB_TOKEN ) then
  echo "Using token from HUGGINGFACE_HUB_TOKEN for login."
  python -c "import os; from huggingface_hub import login; token=os.environ.get(\"HUGGINGFACE_HUB_TOKEN\") or os.environ.get(\"HF_TOKEN\");\nimport sys;\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nif not token: raise SystemExit(\"Missing HUGGINGFACE_HUB_TOKEN/HF_TOKEN; cannot login non-interactively.\"); login(token=token, add_to_git_credential=True); print(\"Login succeeded.\")"
else
  echo "No HUGGINGFACE_HUB_TOKEN provided; attempting download without login (may fail if gated)."
endif

python "$repo/scripts/download_mer2023.py"

if ($status != 0) then
  echo "Download failed. If access is gated, set your Hugging Face token first:"
  echo "  setenv HUGGINGFACE_HUB_TOKEN <your_hf_token>"
  echo "Then re-run: scripts/download_mer2023.csh"
  exit $status
endif

echo "MER2023 downloaded to: $target"
exit 0
