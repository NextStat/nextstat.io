"""Allow ``python -m nextstat`` to invoke the CLI binary."""

from __future__ import annotations

import os
import sys
import sysconfig


def _find_cli() -> str | None:
    """Locate the ``nextstat`` binary installed by the *nextstat-cli* wheel."""
    scripts = sysconfig.get_path("scripts")
    if scripts:
        candidate = os.path.join(scripts, "nextstat")
        if os.path.isfile(candidate):
            return candidate
        # Windows: nextstat.exe
        candidate_exe = candidate + ".exe"
        if os.path.isfile(candidate_exe):
            return candidate_exe
    return None


def main() -> None:
    cli = _find_cli()
    if cli is None:
        print(
            "error: nextstat CLI binary not found.\n"
            "Install it with:  pip install nextstat-cli",
            file=sys.stderr,
        )
        raise SystemExit(1)

    argv = [cli, *sys.argv[1:]]

    if sys.platform != "win32":
        os.execv(cli, argv)
    else:
        import subprocess

        raise SystemExit(subprocess.call(argv))


if __name__ == "__main__":
    main()
