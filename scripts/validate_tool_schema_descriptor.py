"""Validate the tool discovery descriptor against the formal schema.

Usage:
  python scripts/validate_tool_schema_descriptor.py
  python scripts/validate_tool_schema_descriptor.py --transport server --server-url http://127.0.0.1:3742
  python scripts/validate_tool_schema_descriptor.py --transport server --server-url http://127.0.0.1:3742 --api-key secret
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _schema_path() -> Path:
    return _repo_root() / "docs" / "schemas" / "tools" / "nextstat_tool_schema_v1.schema.json"


def _repo_python_root() -> Path:
    return _repo_root() / "bindings" / "ns-py" / "python"


def _activate_repo_python_package() -> None:
    repo_root = str(_repo_python_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import nextstat

    repo_pkg = _repo_python_root() / "nextstat"
    pkg_path = str(repo_pkg)
    package_paths = getattr(nextstat, "__path__", None)
    if package_paths is None:
        raise RuntimeError("nextstat package has no __path__; cannot overlay repo Python modules")
    if pkg_path in package_paths:
        package_paths.remove(pkg_path)
    package_paths.insert(0, pkg_path)
    sys.modules.pop("nextstat.tools", None)
    sys.modules.pop("nextstat._tool_manifest", None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["local", "server"], default="local")
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args(argv)

    try:
        import jsonschema  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("jsonschema is required to validate the tool schema descriptor") from exc

    _activate_repo_python_package()

    from nextstat.tools import get_toolkit_descriptor

    descriptor = get_toolkit_descriptor(
        transport=args.transport,
        server_url=args.server_url,
        api_key=args.api_key,
        timeout_s=args.timeout_s,
    )
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))
    jsonschema.validate(instance=descriptor, schema=schema)
    print(f"Validated {args.transport} tool descriptor against {_schema_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
