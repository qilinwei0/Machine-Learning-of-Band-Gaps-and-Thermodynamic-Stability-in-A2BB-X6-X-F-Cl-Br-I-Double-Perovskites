#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash run_v9_ml.sh INPUT_DIR OUTPUT_ROOT" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$script_dir/generate_v9_ml_data.py" --input-dir "$1" --output-root "$2"
python "$script_dir/validate_v9_data.py" --package-root "$2"

