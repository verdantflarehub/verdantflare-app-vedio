"""Read-only real Ref2VA weight slices; compare staged fusion with sealed upstream.

Creates only temporary tensor slices, never writes into the shared model mount.
This is a component comparison, not generation or creative acceptance.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile

from gpu_guard import verify_gpu_allocation
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from lora_staging import load


def skeleton(name, weight):
    root = torch.nn.Module()
    node = root
    *parents, leaf = name.split('.')
    for part in parents:
        child = torch.nn.Module()
        node.add_module(part, child)
        node = child
    # Avoid allocating and initializing an extra large weight.
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device='meta')
    linear.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    node.add_module(leaf, linear)
    return root


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--files', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    args = p.parse_args()
    entry = json.loads((args.files / 'series.json').read_text())['patches'][0]
    original = load('real_original', args.files / 'original.py', entry['before_sha256'])
    staged = load('real_staged', args.files / 'staged.py', entry['after_sha256'])
    model = args.models / 'MiniMax-H3-Diffusers'
    adapter = args.models / 'MiniMax-H3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors'
    with adapter.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == '9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99'
    index = json.loads((model / 'transformer_ref/diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    assert torch.cuda.device_count() == 2
    uuids = [str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-') for i in range(2)]
    assert len(set(uuids)) == 2
    verify_gpu_allocation(uuids)
    results = []
    with safe_open(adapter, framework='pt', device='cpu') as lora, tempfile.TemporaryDirectory() as tmp:
        pairs, diffs = original._payload(list(lora.keys()), adapter, hybrid=False)
        assert not diffs
        sizes = {name:lora.get_slice(b).get_shape()[0] * lora.get_slice(a).get_shape()[1] for name,(a,b) in pairs.items()}
        selected = [min(sizes, key=lambda n:(sizes[n],n)), max(sizes, key=lambda n:(sizes[n],n))]
        for i in range(2):
            device = torch.device(f'cuda:{i}')
            prop = torch.cuda.get_device_properties(i)
            assert '4090' in prop.name and (prop.major, prop.minor) == (8,9)
            assert prop.total_memory >= 23 * 2**30
            with torch.cuda.device(device):
                for name in selected:
                    a,b = pairs[name]
                    key = name + '.weight'
                    shard = model / 'transformer_ref' / index[key]
                    assert shard.resolve().is_relative_to(model.resolve())
                    with safe_open(shard, framework='pt', device='cpu') as base:
                        weight = base.get_tensor(key)
                    path = Path(tmp) / 'slice.safetensors'
                    save_file({a:lora.get_tensor(a),b:lora.get_tensor(b)}, str(path))
                    resident = skeleton(name, weight).to(device)
                    original.fuse_lora(resident, path, alpha=8)
                    expected = resident.get_submodule(name).weight.detach().cpu()
                    del resident
                    torch.cuda.empty_cache()
                    candidate = skeleton(name, weight)
                    staged.fuse_lora(candidate, path, alpha=8, staging_device=device)
                    actual = candidate.get_submodule(name).weight.detach()
                    assert actual.device.type == 'cpu'
                    assert torch.equal(expected, actual), name
                    result = {'gpu':i,'target':name,'shape':list(weight.shape),'dtype':str(weight.dtype),
                              'base_tensor_sha256':hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest(),
                              'bitwise_equal_to_upstream':True}
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    del candidate, expected, actual, weight
                    gc.collect()
    print(json.dumps({'status':'PASS','scope':'real weight slices only; full base revision not yet verified',
                      'torch':torch.__version__,'uuid':uuids,'results':results}), flush=True)


if __name__ == '__main__':
    main()
