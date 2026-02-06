#!/usr/bin/env python3
"""Build a versioned White Paper PDF from Markdown.

This is intentionally dependency-light:
- Default path uses Docker to run pandoc+LaTeX (no local pandoc install needed).
- Preprocesses Mermaid fenced blocks into a PDF-friendly representation.

Outputs (gitignored):
- dist/whitepaper/nextstat-whitepaper-v<version>.pdf
- dist/whitepaper/nextstat-whitepaper-v<version>.pdf.sha256
- dist/whitepaper/nextstat-whitepaper-latest.pdf
- dist/whitepaper/nextstat-whitepaper-latest.pdf.sha256
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_workspace_version(cargo_toml: Path) -> str:
    """Parse version from `[workspace.package]` in the root Cargo.toml."""
    in_block = False
    for raw in cargo_toml.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_block = line == "[workspace.package]"
            continue
        if in_block:
            m = re.match(r'version\s*=\s*"([^"]+)"\s*$', line)
            if m:
                return m.group(1)
    raise RuntimeError("Failed to parse version from Cargo.toml ([workspace.package].version)")


def _rewrite_mermaid_fences(md: str) -> Tuple[str, int]:
    """Replace ```mermaid fences with a PDF-friendly representation.

    We keep the mermaid source (useful for the PDF reader) but avoid relying on
    Mermaid rendering during PDF build.
    """
    lines = md.splitlines()
    out: List[str] = []
    i = 0
    n_blocks = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```mermaid"):
            n_blocks += 1
            out.append("Diagram (Mermaid source; rendered in the Markdown version):")
            out.append("")
            out.append("```text")
            i += 1
            while i < len(lines) and lines[i].strip() != "```":
                out.append(lines[i])
                i += 1
            out.append("```")
            # Consume closing fence if present.
            if i < len(lines) and lines[i].strip() == "```":
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + ("\n" if md.endswith("\n") else ""), n_blocks


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def _docker_pandoc(
    *,
    repo: Path,
    docker_image: str,
    in_md_rel: Path,
    out_pdf_rel: Path,
    extra_args: List[str],
) -> None:
    cmd = ["docker", "run", "--rm"]
    try:
        uid = os.getuid()
        gid = os.getgid()
        cmd += ["--user", f"{uid}:{gid}"]
    except AttributeError:
        pass
    cmd += ["-v", f"{repo}:/data", "-w", "/data", docker_image]
    cmd += [
        "pandoc",
        str(in_md_rel),
        "-o",
        str(out_pdf_rel),
        "--from=gfm",
        "--toc",
        "--pdf-engine=xelatex",
        "-V",
        "geometry:margin=1in",
    ]
    cmd += extra_args
    _run(cmd)


def _local_pandoc(*, in_md: Path, out_pdf: Path, extra_args: List[str]) -> None:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("pandoc not found; install pandoc or run without --no-docker")
    cmd = [
        pandoc,
        str(in_md),
        "-o",
        str(out_pdf),
        "--from=gfm",
        "--toc",
        "--pdf-engine=xelatex",
        "-V",
        "geometry:margin=1in",
    ]
    cmd += extra_args
    _run(cmd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--version",
        type=str,
        default=None,
        help="Whitepaper version for output filename (default: Cargo.toml workspace version).",
    )
    ap.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("docs/WHITEPAPER.md"),
        help="Input Markdown path (default: docs/WHITEPAPER.md).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dist/whitepaper"),
        help="Output directory for PDFs (default: dist/whitepaper).",
    )
    ap.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("tmp/whitepaper_build"),
        help="Temporary build dir inside repo (default: tmp/whitepaper_build).",
    )
    ap.add_argument(
        "--docker-image",
        type=str,
        default="pandoc/latex:latest",
        help="Docker image to use for pandoc+LaTeX (default: pandoc/latex:latest).",
    )
    ap.add_argument(
        "--no-docker",
        action="store_true",
        help="Use local pandoc instead of Docker (requires pandoc + LaTeX installed).",
    )
    args, unknown = ap.parse_known_args()

    repo = _repo_root()
    cargo_toml = repo / "Cargo.toml"
    version = args.version or _read_workspace_version(cargo_toml)

    in_md = (repo / args.in_path).resolve()
    if not in_md.exists():
        print(f"Missing input: {in_md}", file=sys.stderr)
        return 2

    out_dir = (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = (repo / args.tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pre_md = tmp_dir / "WHITEPAPER.preprocessed.md"
    md_in = in_md.read_text(encoding="utf-8")
    md_out, n_mermaid = _rewrite_mermaid_fences(md_in)
    pre_md.write_text(md_out, encoding="utf-8")

    out_pdf = out_dir / f"nextstat-whitepaper-v{version}.pdf"
    out_latest = out_dir / "nextstat-whitepaper-latest.pdf"

    try:
        if args.no_docker:
            _local_pandoc(in_md=pre_md, out_pdf=out_pdf, extra_args=unknown)
        else:
            in_rel = pre_md.relative_to(repo)
            out_rel = out_pdf.relative_to(repo)
            _docker_pandoc(
                repo=repo,
                docker_image=str(args.docker_image),
                in_md_rel=in_rel,
                out_pdf_rel=out_rel,
                extra_args=unknown,
            )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 3

    shutil.copy2(out_pdf, out_latest)
    (out_pdf.with_suffix(out_pdf.suffix + ".sha256")).write_text(_sha256(out_pdf) + "\n")
    (out_latest.with_suffix(out_latest.suffix + ".sha256")).write_text(_sha256(out_latest) + "\n")

    if n_mermaid:
        print(f"note: converted {n_mermaid} mermaid fenced block(s) to text")
    print(f"wrote: {out_pdf}")
    print(f"wrote: {out_latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

