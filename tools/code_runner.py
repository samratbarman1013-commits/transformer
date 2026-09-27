"""Run a short Python snippet in a subprocess.

Best-effort isolation for the PROTOTYPE only: a fresh python process, a
filesystem-wide timeout, and a blocklist of obviously destructive commands.
Real deployments need a proper sandbox (containers, nsjail/gVisor, or a
managed executor) before untrusted code ever runs.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from .registry import ToolSpec

SPEC = ToolSpec(
    name="run_python",
    description="Run a short Python snippet (no third-party imports, no network) and return stdout/stderr.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "the Python code to run"},
            "timeout": {"type": "integer", "description": "max seconds to run (default 5, max 10)"},
        },
        "required": ["code"],
    },
)

MAX_TIMEOUT = 10

BLOCKED_PATTERNS = (
    "rm -rf", "sudo ", "sudo(", "shutdown", "reboot", "mkfs", "dd if=",
    "os.system", "subprocess", "socket.socket", "urllib", "requests",
)


def _is_blocked(code: str) -> str | None:
    lowered = code.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lowered:
            return pattern.strip()
    return None


def run_python(code: str, timeout: int = 5) -> dict:
    timeout = max(1, min(timeout, MAX_TIMEOUT))
    blocked = _is_blocked(code)
    if blocked:
        return {"ok": False, "error": f"refused: blocked pattern '{blocked}' in code"}

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],  # -I: isolated mode, no user site-packages
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {"ok": proc.returncode == 0, "stdout": proc.stdout[:4000], "stderr": proc.stderr[:4000]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"execution exceeded {timeout}s and was killed"}
