import subprocess


def test_validation_pack_script_bash_syntax_ok() -> None:
    # Keep this lightweight and platform-agnostic: just ensure the script parses.
    subprocess.run(["bash", "-n", "validation-pack/render_validation_pack.sh"], check=True)


def test_validation_pack_script_help_mentions_json_only() -> None:
    out = subprocess.run(
        ["bash", "validation-pack/render_validation_pack.sh", "--help"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    assert "--json-only" in out
    # Signing flags are part of the validation-pack contract (manifest distribution workflow).
    assert "--sign-gpg" in out
    assert "--sign-openssl-key" in out
