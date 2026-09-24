"""Standalone raw-folder pre-pipeline audit (architecture §8).

Runs BEFORE any pipeline stage: point at a freshly-extracted tree, get a
fast "is this raw data intact" verdict. No pipeline state, no ``proced/``
imports — pairing/domain logic is an independent second implementation.
"""

from .run_raw_folder_audit import run_audit  # noqa: F401
