"""Process-wide tuning hook for ROCm containers.

Loaded automatically by Python if present as `sitecustomize.py`.
"""

from __future__ import annotations

import os

# Conservative default to reduce fork-related instability in containerized runs.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
