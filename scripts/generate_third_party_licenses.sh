#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${REPO_ROOT}/THIRD_PARTY_LICENSES"

# In sandboxed environments, keep Cargo state inside the repo if possible.
if [[ -d "${REPO_ROOT}/.cargo-home" && -z "${CARGO_HOME:-}" ]]; then
  export CARGO_HOME="${REPO_ROOT}/.cargo-home"
fi
if [[ -d "${REPO_ROOT}/.rustup" && -z "${RUSTUP_HOME:-}" ]]; then
  export RUSTUP_HOME="${REPO_ROOT}/.rustup"
fi

tmp_meta="$(mktemp)"
trap 'rm -f "$tmp_meta"' EXIT

cd "${REPO_ROOT}"
cargo metadata --format-version 1 > "${tmp_meta}"

{
  echo "# Third-party licenses"
  echo
  echo "Generated (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')"
  echo

  echo "## Rust (Cargo)"
  python3 - "${tmp_meta}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for p in data.get("packages", []):
    # Workspace packages have no "source"; only list third-party deps.
    if p.get("source") is None:
        continue
    name = p.get("name") or "UNKNOWN"
    ver = p.get("version") or "UNKNOWN"
    lic = p.get("license") or "UNKNOWN"
    rows.append((name, ver, lic))

for name, ver, lic in sorted(set(rows), key=lambda t: (t[0].lower(), t[1])):
    print(f"- {name} {ver} — {lic}")
PY
  echo

  echo "## Python (PyPI metadata; bindings/ns-py/pyproject.toml)"
  python3 - <<'PY'
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

import tomllib


REPO_ROOT = Path.cwd()
PYPROJECT = REPO_ROOT / "bindings" / "ns-py" / "pyproject.toml"

name_re = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def req_name(req: str) -> str:
    m = name_re.match(req)
    return (m.group(1) if m else req).lower()


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "nextstat-license-report"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def pypi_license(info: dict) -> str:
    lic = (info.get("license") or "").strip()
    if lic:
        return lic
    classifiers = info.get("classifiers") or []
    lic_cls = [c for c in classifiers if c.startswith("License ::")]
    if lic_cls:
        # Keep the full classifier(s) to avoid lossy parsing.
        return "; ".join(lic_cls)
    return "UNKNOWN"


py = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
reqs: list[str] = []
reqs.extend((py.get("build-system") or {}).get("requires") or [])
reqs.extend((py.get("project") or {}).get("dependencies") or [])
for group in ((py.get("project") or {}).get("optional-dependencies") or {}).values():
    reqs.extend(group or [])

pkgs = sorted({req_name(r) for r in reqs if r})

rows = []
for pkg in pkgs:
    try:
        data = fetch_json(f"https://pypi.org/pypi/{pkg}/json")
        info = data.get("info") or {}
        ver = info.get("version") or "UNKNOWN"
        lic = pypi_license(info)
        rows.append((pkg, ver, lic))
    except Exception as e:
        rows.append((pkg, "ERR", f"ERR({e.__class__.__name__})"))

for pkg, ver, lic in rows:
    print(f"- {pkg} {ver} — {lic}")
PY
} > "${OUT}"

echo "Wrote ${OUT}" >&2
