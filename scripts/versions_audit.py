#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "nextstat-versions-audit"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


@dataclass(frozen=True)
class Semver:
    major: int
    minor: int
    patch: int
    prerelease: str | None = None

    @staticmethod
    def parse(version: str) -> "Semver":
        v = version.strip()
        if v.startswith("v"):
            v = v[1:]
        v = v.split("+", 1)[0]
        main, sep, pre = v.partition("-")
        parts = main.split(".")
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        prerelease = pre if sep else None
        return Semver(major=major, minor=minor, patch=patch, prerelease=prerelease)

    def without_prerelease(self) -> "Semver":
        return Semver(self.major, self.minor, self.patch, None)


def cargo_req_covers_latest(req: str, latest: Semver) -> bool:
    """Best-effort check for plain Cargo version strings (caret requirements).

    Examples:
    - "1.0" means >=1.0.0,<2.0.0
    - "0.17" means >=0.17.0,<0.18.0
    """
    m = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?\s*$", req)
    if not m:
        return True  # Unknown syntax; don't flag.
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    patch = int(m.group(3) or 0)

    lower = Semver(major, minor, patch)
    if major == 0:
        upper = Semver(0, minor + 1, 0)
    else:
        upper = Semver(major + 1, 0, 0)

    latest_stable = latest.without_prerelease()
    return (lower.major, lower.minor, lower.patch) <= (
        latest_stable.major,
        latest_stable.minor,
        latest_stable.patch,
    ) < (upper.major, upper.minor, upper.patch)


def cratesio_latest_versions(crate: str) -> tuple[str, str]:
    """Returns (max_version, latest_stable_version)."""
    info = fetch_json(f"https://crates.io/api/v1/crates/{crate}")
    max_version = info["crate"]["max_version"]
    if "-" not in max_version:
        return max_version, max_version

    versions = fetch_json(f"https://crates.io/api/v1/crates/{crate}/versions?per_page=100")[
        "versions"
    ]
    for v in versions:
        num = v.get("num", "")
        if num and "-" not in num:
            return max_version, num
    return max_version, max_version


def pypi_latest_version(pkg: str) -> str:
    data = fetch_json(f"https://pypi.org/pypi/{pkg}/json")
    return data["info"]["version"]


def parse_python_req_name(req: str) -> str:
    # "maturin>=1.11,<2.0" -> "maturin"
    m = re.match(r"^\s*([A-Za-z0-9_.-]+)", req)
    return (m.group(1) if m else req).lower()


def github_latest_tag(owner: str, repo: str) -> str | None:
    # Prefer semver-ish tags (typical for GitHub Actions). Avoid "bundle" releases.
    headers = {"User-Agent": "nextstat-versions-audit"}
    semver_tag_re = re.compile(r"^v\d+(?:\.\d+){0,2}$")

    def is_semver_tag(tag: str) -> bool:
        return bool(semver_tag_re.match(tag.strip()))

    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{owner}/{repo}/releases/latest", headers=headers
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        tag = data.get("tag_name")
        if tag and is_semver_tag(tag):
            return tag
    except Exception:
        pass

    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{owner}/{repo}/tags?per_page=20", headers=headers
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        if data:
            for item in data:
                name = (item.get("name") or "").strip()
                if name and is_semver_tag(name):
                    return name
            return (data[0].get("name") or "").strip() or None
    except Exception:
        pass
    return None


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit NextStat dependency/tooling versions.")
    parser.add_argument(
        "--fail-on-outdated",
        action="store_true",
        help="Exit non-zero if any Rust crate constraint doesn't cover the latest stable release.",
    )
    args = parser.parse_args()

    now = dt.datetime.now(dt.timezone.utc).astimezone()

    cargo_toml = load_toml(REPO_ROOT / "Cargo.toml")
    ws_deps = (cargo_toml.get("workspace") or {}).get("dependencies") or {}

    rust_toolchain = None
    toolchain_path = REPO_ROOT / "rust-toolchain.toml"
    if toolchain_path.exists():
        toolchain = load_toml(toolchain_path).get("toolchain") or {}
        rust_toolchain = toolchain.get("channel")

    pyproject_path = REPO_ROOT / "bindings" / "ns-py" / "pyproject.toml"
    pyproject = load_toml(pyproject_path)
    requires_python = (pyproject.get("project") or {}).get("requires-python")
    py_reqs: list[str] = []
    py_reqs.extend((pyproject.get("build-system") or {}).get("requires") or [])
    py_reqs.extend((pyproject.get("project") or {}).get("dependencies") or [])
    for group in ((pyproject.get("project") or {}).get("optional-dependencies") or {}).values():
        py_reqs.extend(group or [])
    py_pkgs = sorted({parse_python_req_name(r) for r in py_reqs if r})

    workflow_uses: dict[str, set[str]] = {}
    uses_re = re.compile(r"^\s*uses:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([^\s#]+)")
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    if workflows_dir.exists():
        for wf in sorted(workflows_dir.glob("*.yml")):
            for line in wf.read_text(encoding="utf-8").splitlines():
                m = uses_re.match(line)
                if not m:
                    continue
                action = m.group(1)
                ref = m.group(2)
                workflow_uses.setdefault(action, set()).add(ref)

    print("# Version Audit Snapshot\n")
    print(f"- Generated: {now.strftime('%Y-%m-%d %H:%M:%S %z')}")
    if rust_toolchain:
        print(f"- Rust toolchain (pinned): `{rust_toolchain}`")
    if requires_python:
        print(f"- Python requires: `{requires_python}`")
    print()

    print("## Rust crates (workspace.dependencies)\n")
    print("| Crate | Constraint | Latest (stable) | Status |")
    print("|---|---:|---:|---|")

    outdated = []
    for crate, spec in sorted(ws_deps.items(), key=lambda kv: kv[0].lower()):
        if isinstance(spec, str):
            req = spec
        elif isinstance(spec, dict):
            req = str(spec.get("version") or "")
            if not req:
                continue
        else:
            continue

        try:
            max_v, stable_v = cratesio_latest_versions(crate)
            latest = Semver.parse(stable_v)
            ok = cargo_req_covers_latest(req, latest)
            status = "OK" if ok else "OUTDATED"
            if not ok:
                outdated.append(crate)
        except Exception as e:
            stable_v = f"ERR({e.__class__.__name__})"
            status = "ERR"

        print(f"| `{crate}` | `{req}` | `{stable_v}` | {status} |")

    print()
    print("## Python packages (bindings/ns-py/pyproject.toml)\n")
    print("| Package | Latest |")
    print("|---|---:|")
    for pkg in py_pkgs:
        try:
            latest = pypi_latest_version(pkg)
        except Exception as e:
            latest = f"ERR({e.__class__.__name__})"
        print(f"| `{pkg}` | `{latest}` |")

    print()
    print("## GitHub Actions pins (.github/workflows/*.yml)\n")
    print("| Action | Current refs | Latest tag |")
    print("|---|---|---:|")
    for action, refs in sorted(workflow_uses.items()):
        owner, repo = action.split("/", 1)
        latest = github_latest_tag(owner, repo) or "unknown"
        current = ", ".join(f"`{r}`" for r in sorted(refs))
        print(f"| `{action}` | {current} | `{latest}` |")

    print()
    if outdated:
        print("## Outdated Rust constraints\n")
        print(
            "These `Cargo.toml` constraints do not cover the latest stable release on crates.io:"
        )
        for crate in outdated:
            print(f"- `{crate}`")
        print()

    if args.fail_on_outdated and outdated:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
