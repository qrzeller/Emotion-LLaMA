#!/usr/bin/env tcsh
# Small helper to run Emotion-LLaMA evaluation with the repo's virtualenv.
# Ensures the venv torchrun and packages (e.g., tqdm) are used and avoids PATH/rehash pitfalls.

# Resolve script directory robustly (works regardless of current dir)
set real0 = `which $0`
set scriptdir = `dirname "$real0"`
cd "$scriptdir/.." || exit 1
set repo = "$cwd"

if (! -e "$repo/.venv/bin/activate.csh") then
  echo "Error: .venv not found at $repo/.venv. Create it and install requirements first."
  echo "For example (tcsh):"
  echo "  python3 -m venv .venv && source .venv/bin/activate.csh && rehash && pip install -r requirements.txt"
  exit 1
endif

source "$repo/.venv/bin/activate.csh"
rehash

# Defaults (can be overridden via args below)
set nproc = 1
set cfg = "eval_configs/eval_emotion.yaml"
set dataset = "feature_face_caption"

# Simple flag parsing for common options; unknown args are forwarded.
set pass_args = ()
set i = 1
while ($i <= $#argv)
  switch ($argv[$i])
    case --cfg-path:
      @ i++
      if ($i <= $#argv) set cfg = "$argv[$i]"
      breaksw
    case --dataset:
      @ i++
      if ($i <= $#argv) set dataset = "$argv[$i]"
      breaksw
    case --nproc-per-node:
      @ i++
      if ($i <= $#argv) set nproc = "$argv[$i]"
      breaksw
    default:
      set pass_args = ($pass_args "$argv[$i]")
  endsw
  @ i++
end

echo "Using venv: `which python`"
echo "torchrun: `which torchrun`"

"$repo/.venv/bin/torchrun" --nproc-per-node $nproc \
  "$repo/eval_emotion.py" --cfg-path "$cfg" --dataset "$dataset" $pass_args
