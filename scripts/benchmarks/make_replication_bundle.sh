#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
make_replication_bundle.sh

Create a publishable replication bundle for a rerun artifact directory.

Writes into: <out-dir>/.replication/

Outputs:
  - .replication/original_snapshot_index.json
  - snapshot_index.json                         (hash index for <out-dir>, excluding .replication/)
  - .replication/replication_report.json
  - .replication/replication_report.sha256(.bin)
  - .replication/replication_report.json.sig    (if --sign-openssl-key)
  - .replication/README.md

Usage:
  bash scripts/benchmarks/make_replication_bundle.sh [options]

Options:
  --original-index PATH         Required: published snapshot_index.json
  --artifacts-dir DIR           Required: rerun artifacts directory
  --out-dir DIR                 Optional: output directory (default: --artifacts-dir)
  --suite NAME                  Optional: suite name for the replica index (default: replication)
  --snapshot-id ID              Optional: snapshot id for the replica index
  --notes TEXT                  Optional: short notes for replication_report.json
  --sign-openssl-key PATH       Optional: sign replication_report.json digest with OpenSSL (pkeyutl)
  --sign-openssl-pub PATH       Optional: copy public key to .replication/replication_report.pub.pem
  -h, --help                    Show this help
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

original_index=""
artifacts_dir=""
out_dir=""
suite="replication"
snapshot_id=""
notes=""
openssl_key=""
openssl_pub=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --original-index) original_index="$2"; shift 2 ;;
    --artifacts-dir) artifacts_dir="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --suite) suite="$2"; shift 2 ;;
    --snapshot-id) snapshot_id="$2"; shift 2 ;;
    --notes) notes="$2"; shift 2 ;;
    --sign-openssl-key) openssl_key="$2"; shift 2 ;;
    --sign-openssl-pub) openssl_pub="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$original_index" || -z "$artifacts_dir" ]]; then
  echo "Missing required args: --original-index and --artifacts-dir" >&2
  usage >&2
  exit 2
fi

if [[ -z "$out_dir" ]]; then
  out_dir="$artifacts_dir"
fi

if [[ "$original_index" != /* ]]; then
  original_index="$repo_root/$original_index"
fi
if [[ "$artifacts_dir" != /* ]]; then
  artifacts_dir="$repo_root/$artifacts_dir"
fi
if [[ "$out_dir" != /* ]]; then
  out_dir="$repo_root/$out_dir"
fi

if [[ ! -f "$original_index" ]]; then
  echo "Original snapshot_index.json not found: $original_index" >&2
  exit 2
fi
if [[ ! -d "$artifacts_dir" ]]; then
  echo "Artifacts dir not found: $artifacts_dir" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Missing dependency: python3" >&2
  exit 2
fi

mkdir -p "$out_dir"
rep_dir="$out_dir/.replication"
mkdir -p "$rep_dir"

cp "$original_index" "$rep_dir/original_snapshot_index.json"

replica_index="$out_dir/snapshot_index.json"
python3 "$repo_root/scripts/benchmarks/write_snapshot_index.py" \
  --suite "$suite" \
  --artifacts-dir "$artifacts_dir" \
  --out "$replica_index" \
  ${snapshot_id:+--snapshot-id "$snapshot_id"}

rep_report="$rep_dir/replication_report.json"
python3 "$repo_root/scripts/benchmarks/write_replication_report.py" \
  --original-index "$rep_dir/original_snapshot_index.json" \
  --replica-index "$replica_index" \
  --out "$rep_report" \
  ${notes:+--notes "$notes"}

rep_sha_hex="$rep_dir/replication_report.sha256"
rep_sha_bin="$rep_dir/replication_report.sha256.bin"
python3 - "$rep_report" "$rep_sha_hex" "$rep_sha_bin" <<'PY'
import hashlib
import sys
from pathlib import Path

inp = Path(sys.argv[1])
out_hex = Path(sys.argv[2])
out_bin = Path(sys.argv[3])

h = hashlib.sha256(inp.read_bytes()).hexdigest()
out_hex.write_text(h + "\n", encoding="utf-8")
out_bin.write_bytes(bytes.fromhex(h))
PY

rep_sig="$rep_dir/replication_report.json.sig"
rep_pub="$rep_dir/replication_report.pub.pem"
if [[ -n "$openssl_key" ]]; then
  if [[ ! -f "$openssl_key" ]]; then
    echo "OpenSSL key not found: $openssl_key" >&2
    exit 2
  fi
  if ! command -v openssl >/dev/null 2>&1; then
    echo "Missing dependency: openssl (required for --sign-openssl-key)" >&2
    exit 2
  fi
  openssl pkeyutl -sign -inkey "$openssl_key" -rawin -in "$rep_sha_bin" -out "$rep_sig"
  if [[ -n "$openssl_pub" ]]; then
    if [[ ! -f "$openssl_pub" ]]; then
      echo "OpenSSL public key not found: $openssl_pub" >&2
      exit 2
    fi
    cp "$openssl_pub" "$rep_pub"
  fi
fi

cat >"$rep_dir/README.md" <<EOF
# Replication Bundle

Files:
- \`snapshot_index.json\` (rerun artifact hashes; excludes \`.replication/\`)
- \`.replication/original_snapshot_index.json\`
- \`.replication/replication_report.json\`

Schemas:
- \`docs/schemas/benchmarks/snapshot_index_v1.schema.json\`
- \`docs/schemas/benchmarks/replication_report_v1.schema.json\`

Verify replication_report.json digest:

\`\`\`bash
cat .replication/replication_report.sha256
\`\`\`

If signed (OpenSSL):

\`\`\`bash
openssl pkeyutl -verify -pubin -inkey .replication/replication_report.pub.pem -rawin \\
  -in .replication/replication_report.sha256.bin \\
  -sigfile .replication/replication_report.json.sig
\`\`\`
EOF

echo "Wrote:" >&2
echo "  $replica_index" >&2
echo "  $rep_report" >&2
echo "  $rep_sha_hex" >&2
echo "  $rep_sha_bin" >&2
if [[ -n "$openssl_key" ]]; then
  echo "  $rep_sig" >&2
  if [[ -n "$openssl_pub" ]]; then
    echo "  $rep_pub" >&2
  fi
fi
echo "  $rep_dir/README.md" >&2

