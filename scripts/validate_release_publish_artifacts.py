from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


def _collect_package_files(dist_dir: Path, package_prefix: str) -> list[Path]:
    files = sorted(dist_dir.glob(f"{package_prefix}-*.whl"))
    files.extend(sorted(dist_dir.glob(f"{package_prefix}-*.tar.gz")))
    if not files:
        raise SystemExit(
            f"no release publish artifacts found for prefix {package_prefix!r} in {dist_dir}"
        )
    return files


def _validate_file(path: Path) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise SystemExit(f"{path} is a corrupt wheel; first bad member: {bad_member}")
        return

    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            archive.getmembers()
        return

    raise SystemExit(f"unsupported publish artifact: {path}")


def validate_release_publish_artifacts(dist_dir: Path, package_prefix: str) -> list[Path]:
    files = _collect_package_files(dist_dir, package_prefix)
    for path in files:
        _validate_file(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--package-prefix", required=True)
    args = parser.parse_args()

    files = validate_release_publish_artifacts(
        dist_dir=args.dir.resolve(),
        package_prefix=args.package_prefix,
    )
    print(
        "validated",
        f"package_prefix={args.package_prefix}",
        f"count={len(files)}",
        *[path.name for path in files],
    )


if __name__ == "__main__":
    main()
