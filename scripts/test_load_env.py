#!/usr/bin/env python
"""Tests for scripts/load_env.sh and the sweep scripts' use of .env.

  1. load_env.sh on a hand-written env file: KEY=VALUE lines are exported, comments,
     blank and malformed lines are skipped, an optional "export " and quotes around the
     value are accepted, CRLF line ends are tolerated, and a variable that is already
     set is never overridden. A missing file is not an error.
  2. Each sweep script, run against a throwaway repo directory (REPO=...) with an empty
     job list so nothing is launched: with no key anywhere it refuses to start; with the
     key in <repo>/.env it starts; with the key exported and no .env it starts.

Every subprocess runs with a minimal environment, so no key of the caller leaks in.

    python scripts/test_load_env.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name:<70s} {detail}")
    if not cond:
        FAILS.append(name)


def bash(script: str, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp")}
    return subprocess.run(["bash", "-c", script, "_", *args], env=base | (env or {}),
                          capture_output=True, text=True, timeout=120)


def test_loader(tmp: Path) -> None:
    print("\n1. load_env.sh")
    f = tmp / "test.env"
    f.write_bytes(b"# a comment\n"
                  b"\n"
                  b"A=plain\n"
                  b'export B="two words"\n'
                  b"C='single'\n"
                  b"E=\n"
                  b"F=with=equals\n"
                  b"G=crlf\r\n"
                  b"PRESET=from-file\n"
                  b"not a valid line\n"
                  b"1BAD=x\n"
                  b"  H=indented")  # no final newline
    show = ('. "$0/load_env.sh" "$1"; '
            'for v in A B C E F G H PRESET; do printf "%s=[%s] " "$v" "${!v-<unset>}"; done; '
            'printf "1BAD lines in env: %s | child sees B=[%s]" "$(env | grep -c "^1BAD=")" "$(printenv B)"')
    r = bash(show.replace("$0", str(SCRIPTS)), str(f), env={"PRESET": "from-env"})
    want = ("A=[plain] B=[two words] C=[single] E=[] F=[with=equals] G=[crlf] H=[indented] "
            "PRESET=[from-env] 1BAD lines in env: 0 | child sees B=[two words]")
    check("values parsed, already-set PRESET kept, malformed lines skipped, exported",
          r.stdout == want and r.stderr == "", f"{r.stdout!r} {r.stderr.strip()!r}")
    r = bash(f'. "{SCRIPTS}/load_env.sh" "$1"; echo ok', str(tmp / "missing.env"))
    check("a missing env file is not an error",
          r.returncode == 0 and r.stdout == "ok\n" and r.stderr == "", f"{r.stdout!r} {r.stderr.strip()!r}")


def test_sweeps(tmp: Path) -> None:
    print("\n2. the sweep scripts read the W&B key from <repo>/.env")
    for script in ("run_gate_valsplit_sweep.sh", "run_gate_sweep.sh"):
        repo = tmp / script.replace(".sh", "")
        repo.mkdir()
        run = f'bash "{SCRIPTS / script}"'
        nojobs = {"REPO": str(repo), "ROW_PERCENTS": " ", "GPUS": "0"}  # " ": no runs at all

        r = bash(run, env=nojobs)
        check(f"{script}: no key anywhere -> refuses to start",
              r.returncode != 0 and ".env" in r.stdout + r.stderr, f"rc {r.returncode}")

        (repo / ".env").write_text("# test\nWANDB_API_KEY=key-from-dotenv\n")
        r = bash(run, env=nojobs)
        check(f"{script}: key in <repo>/.env -> starts (0 runs)",
              r.returncode == 0 and "0 runs" in r.stdout, f"rc {r.returncode} {r.stdout.strip()[-60:]!r}")

        (repo / ".env").unlink()
        r = bash(run, env=nojobs | {"WANDB_API_KEY": "key-from-env"})
        check(f"{script}: key exported, no .env -> starts (0 runs)",
              r.returncode == 0 and "0 runs" in r.stdout, f"rc {r.returncode}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="load_env_test_") as d:
        test_loader(Path(d))
        test_sweeps(Path(d))
    print(f"\n{'ALL PASSED' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
