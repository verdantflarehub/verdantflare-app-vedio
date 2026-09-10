"""Require the current allocation's UUIDs, never a historical node inventory."""
import os
from uuid import UUID


def verify_gpu_allocation(actual):
    def normalize(value):
        return str(UUID(str(value).strip().removeprefix('GPU-')))

    expected = [normalize(value) for value in os.environ.get('SOL_EXPECTED_GPU_UUIDS', '').split(',') if value.strip()]
    visible = [normalize(value) for value in actual]
    if len(expected) != 2 or len(set(expected)) != 2:
        raise ValueError('SOL_EXPECTED_GPU_UUIDS must identify the two approved physical GPUs')
    if len(visible) != 2 or set(visible) != set(expected):
        raise ValueError('visible GPUs differ from the current approved allocation')
    return visible
