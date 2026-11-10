#!/usr/bin/env bash
# Package a trained ensemble as a release asset.
#
# Usage: tools/package_models.sh ENSEMBLE_ID [OUT_DIR]
#
# Writes OUT_DIR/adopt2g-models.tar.gz (default OUT_DIR: dist/) and OUT_DIR/SHA256SUMS.
# The archive unpacks to models/ENSEMBLE_ID/ at the repository root, and carries LICENSE and
# THIRD_PARTY.md inside the ensemble folder, because the fine-tuned ESM-2 weights must keep
# Meta's MIT notice. The build is reproducible: the same ensemble gives the same checksum.
set -euo pipefail

ensemble_id=${1:?usage: tools/package_models.sh ENSEMBLE_ID [OUT_DIR]}
repo=$(cd "$(dirname "$0")/.." && pwd)
out_dir=$(mkdir -p "${2:-$repo/dist}" && cd "${2:-$repo/dist}" && pwd)
src="$repo/models/$ensemble_id"
name=adopt2g-models.tar.gz

[[ -f "$src/manifest.json" ]] || { echo "no manifest at $src/manifest.json" >&2; exit 1; }

# Refuse manifests the current code cannot load, and use the creation time as the archive mtime.
mtime=$(python3 -I - "$src/manifest.json" <<'EOF'
import json, sys
from datetime import datetime
m = json.load(open(sys.argv[1]))
missing = {"temperature_scaling_factor", "uncertainty_calibration_factor"} - set(m)
if missing or "calibration_factor" in m:
    sys.exit(f"manifest uses the pre-temperature-scaling format (missing {sorted(missing)})")
print(int(datetime.fromisoformat(m["created_at"]).timestamp()))
EOF
)

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/models"
cp -a "$src" "$stage/models/"
cp "$repo/LICENSE" "$repo/THIRD_PARTY.md" "$stage/models/$ensemble_id/"

tar --sort=name --owner=0 --group=0 --numeric-owner --mode='u+rwX,go+rX,go-w' --mtime="@$mtime" --format=gnu \
    -C "$stage" -cf - models | gzip -n -9 > "$out_dir/$name"
(cd "$out_dir" && sha256sum "$name" > SHA256SUMS && cat SHA256SUMS)
