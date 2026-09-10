"""CUDA numerical regression with deterministic component fixtures, not video acceptance.

Compare a patched primitive against the sealed upstream on each of two full GPUs.
No model download, generation request, or approval record is fabricated.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from gpu_guard import verify_gpu_allocation
import torch
from safetensors.torch import save_file


def load(name, path, digest):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model():
    torch.manual_seed(20260909)
    m = torch.nn.Module()
    m.blocks = torch.nn.ModuleList()
    for _ in range(4):
        b = torch.nn.Module()
        b.to_q = torch.nn.Linear(2048, 4096, bias=True, dtype=torch.bfloat16)
        m.blocks.append(b)
    return m


def check(original, staged, device, directory, hybrid):
    base = model()
    payload = {}
    for i in range(4):
        prefix = f'diffusion_model.blocks.{i}.to_q'
        payload[prefix + '.lora_A.weight'] = torch.randn(8, 2048, dtype=torch.bfloat16) * .01
        payload[prefix + '.lora_B.weight'] = torch.randn(4096, 8, dtype=torch.bfloat16) * .01
        if hybrid:
            payload[prefix + '.diff_b'] = torch.randn(4096, dtype=torch.float32) * .0001
    # Exercise the large FP32 delta path without turning the fixture into a full H3 model.
    if hybrid:
        payload['diffusion_model.blocks.3.to_q.diff'] = torch.randn(4096, 2048) * .0001
    metadata = {'format': 'fastvideo-lora-v2', 'rank': '8'} if hybrid else None
    path = directory / f'{device.index}-{hybrid}.safetensors'
    save_file(payload, str(path), metadata=metadata)
    del payload
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    resident = copy.deepcopy(base).to(device)
    expected_report = original.fuse_lora(resident, path, alpha=8, scale=.75)
    torch.cuda.synchronize(device)
    resident_peak = torch.cuda.max_memory_allocated(device)
    expected = {k: v.cpu().clone() for k, v in resident.state_dict().items()}
    del resident
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    identities = [id(p) for p in base.parameters()]
    actual_report = staged.fuse_lora(base, path, alpha=8, scale=.75, staging_device=device)
    torch.cuda.synchronize(device)
    staged_peak = torch.cuda.max_memory_allocated(device)
    assert identities == [id(p) for p in base.parameters()], 'Parameter identity changed'
    assert all(p.device.type == 'cpu' for p in base.parameters()), 'CUDA weight leaked'
    for name, actual in base.state_dict().items():
        assert torch.equal(actual, expected[name]), f'numerical mismatch: {name}'
    for key in ('format', 'pairs', 'diffs', 'rank', 'alpha', 'scale', 'effective_scale'):
        assert getattr(actual_report, key) == getattr(expected_report, key), key
    assert staged_peak < resident_peak, (staged_peak, resident_peak)
    # Validation must finish before any base weight is changed.
    bad = directory / 'invalid.safetensors'
    save_file({'blocks.0.to_q.lora_A.weight': torch.ones(8, 2048),
               'blocks.0.to_q.lora_B.weight': torch.ones(4096, 8),
               'blocks.3.to_q.lora_A.weight': torch.ones(8, 2048),
               'blocks.3.to_q.lora_B.weight': torch.ones(4095, 8)}, str(bad))
    before = {k: v.clone() for k, v in base.state_dict().items()}
    try:
        staged.fuse_lora(base, bad, staging_device=device)
        raise AssertionError('invalid shape accepted')
    except ValueError:
        pass
    assert all(torch.equal(v, before[k]) for k, v in base.state_dict().items())
    try:
        staged.fuse_lora(base, path)
        raise AssertionError('default CPU fusion was silently enabled')
    except RuntimeError:
        pass
    try:
        staged.fuse_lora(base, path, staging_device='cpu')
        raise AssertionError('CPU staging accepted')
    except ValueError:
        pass
    return {'device': str(device), 'format': actual_report.format,
            'bitwise_equal_to_upstream': True, 'parameter_identity_preserved': True,
            'rejection_checks': 'pass', 'resident_peak_bytes': resident_peak,
            'staged_peak_bytes': staged_peak}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=Path, required=True)
    args = parser.parse_args()
    series = json.loads((args.files / 'series.json').read_text())['patches'][0]
    original = load('sol_original_lora', args.files / 'original.py', series['before_sha256'])
    staged = load('sol_staged_lora', args.files / 'staged.py', series['after_sha256'])
    assert torch.cuda.device_count() == 2
    uuid = [str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-') for i in range(2)]
    assert len(set(uuid)) == 2
    verify_gpu_allocation(uuid)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(2):
            device = torch.device(f'cuda:{i}')
            p = torch.cuda.get_device_properties(i)
            assert '4090' in p.name and (p.major, p.minor) == (8, 9)
            assert p.total_memory >= 23 * 2**30
            with torch.cuda.device(device):
                for hybrid in (False, True):
                    result = check(original, staged, device, Path(tmp), hybrid)
                    results.append(result)
                    print(json.dumps(result), flush=True)
    print(json.dumps({'status': 'PASS', 'scope': 'LoRA component regression only',
                      'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'uuid': uuid, 'results': results}), flush=True)


if __name__ == '__main__':
    main()
