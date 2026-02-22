"""WSL-friendly entrypoint wrapper for vLLM ROCm images.

It keeps docker-compose commands unchanged by prepending `vllm` to args.
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    # Keep defaults overridable from compose/.env.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    cmd = ["vllm", *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
