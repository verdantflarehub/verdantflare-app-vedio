"""Pinned Sol CUDA regression: staged AdaLN tables then Diffusers block offload.

Technical component fixtures only, not a video request or creative acceptance.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from gpu_guard import verify_gpu_allocation
import torch
from diffusers.hooks import apply_group_offloading
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel


def load(name, file, expected):
    assert hashlib.sha256(file.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Stack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(20260909)
        m = MiniMaxH3Transformer3DModel(
            num_attention_heads=2, attention_head_dim=16, hidden_size=32,
            num_layers=4, num_refiner_layers=1, ffn_dim=64, text_dim=32,
            freq_dim=16, time_embed_hidden_dim=32, time_embed_dim=16, rope_freq_dim=4,
        ).to(dtype=torch.bfloat16)
        self.time_proj = m.time_proj
        self.time_embedder = m.time_embedder.float()
        self.transformer_blocks = m.transformer_blocks

    def forward(self, hidden, temb, indices):
        for block in self.transformer_blocks:
            hidden = block(hidden, temb, indices, None)
        return hidden


@torch.no_grad()
def check(original, staged, device):
    cpu = Stack().eval()
    reference = copy.deepcopy(cpu).to(device)
    # Four steps, including equal/distinct video/audio timesteps and fixed conditions.
    video = torch.tensor([1., .75, .5, .25])
    audio = torch.tensor([1., .6, .3, .1])
    baseline = original.precompute(reference, video, audio, include_audio_condition=True)
    stats = staged.precompute(cpu, video, audio, include_audio_condition=True, staging_device=device)
    assert stats == baseline
    for block, expected in zip(cpu.transformer_blocks, reference.transformer_blocks):
        assert block.adaln_proj.table.device.type == 'cpu'
        assert torch.equal(block.adaln_proj.table, expected.adaln_proj.table.cpu())
    assert all(p.device.type == 'cpu' for p in cpu.parameters())
    # Cache replacement must happen before installing hooks that capture parameters/buffers.
    apply_group_offloading(cpu, onload_device=device, offload_device=torch.device('cpu'),
                          offload_type='block_level', num_blocks_per_group=1,
                          use_stream=False, non_blocking=False)
    torch.manual_seed(99)
    hidden = torch.randn(1, 12, 32, device=device, dtype=torch.bfloat16)
    temb = torch.zeros(4, 16, device=device)
    indices = torch.arange(12, dtype=torch.long, device=device)
    comparisons = 0
    for attempt in range(2):
        for schedule in range(3):
            for step in range(4):
                reference._h3opt_adaln_cursor.set(step, schedule)
                cpu._h3opt_adaln_cursor.set(step, schedule)
                expected = reference(hidden, temb, indices)
                actual = cpu(hidden, temb, indices)
                assert torch.equal(actual, expected), (attempt, schedule, step)
                assert all(p.device.type == 'cpu' for p in cpu.parameters())
                assert all(b.adaln_proj.table.device.type == 'cpu' for b in cpu.transformer_blocks)
                comparisons += 1
    # Validate unsupported staging and malformed schedules before dropping projections.
    fresh = Stack()
    for staging_device, v, a in [('cpu', video, audio), (str(device), video, audio[:-1])]:
        old = [id(b.adaln_proj) for b in fresh.transformer_blocks]
        try:
            staged.precompute(fresh, v, a, staging_device=staging_device)
            raise AssertionError('invalid precompute accepted')
        except ValueError:
            pass
        assert old == [id(b.adaln_proj) for b in fresh.transformer_blocks]
        assert all(p.device.type == 'cpu' for p in fresh.parameters())
    try:
        staged.precompute(cpu, video, audio, staging_device=device)
        raise AssertionError('second precompute accepted')
    except RuntimeError:
        pass
    # Exercise the real denoiser wrapper: replacement precedes hook installation,
    # and later calls do not rebuild caches or attach hooks twice.
    from types import SimpleNamespace
    from diffusers.modular_pipelines.minimax_h3 import denoise
    loop = denoise.MiniMaxH3Ref2VALoopDenoiser
    previous_call = loop.__call__
    previous_marker = getattr(loop, '_h3opt_patched', None)
    fresh = Stack().eval()
    callback_count = []
    def after(component):
        assert all(isinstance(b.adaln_proj, staged.PrecomputedModulation) for b in component.transformer_blocks)
        apply_group_offloading(component, onload_device=device, offload_device=torch.device('cpu'),
                              offload_type='block_level', num_blocks_per_group=1,
                              use_stream=False, non_blocking=False)
        callback_count.append(1)
    def inner(self, components, state, i, t):
        return components.transformer_ref(hidden, temb, indices)
    try:
        loop.__call__ = inner
        if hasattr(loop, '_h3opt_patched'):
            delattr(loop, '_h3opt_patched')
        staged.enable_adaln_precompute(fresh, verbose=False, component_name='transformer_ref',
                                      staging_device=device, after_precompute=after)
        components = SimpleNamespace(transformer_ref=fresh)
        state = SimpleNamespace(timesteps=video, audio_timesteps=audio,
                                num_condition_video_rows=1, num_condition_audio_rows=1)
        for _ in range(2):
            actual = loop()(components, state, 0, video[0])
            reference._h3opt_adaln_cursor.set(0, 2)
            assert torch.equal(actual, reference(hidden, temb, indices))
        assert len(callback_count) == 1
        assert all(p.device.type == 'cpu' for p in fresh.parameters())
    finally:
        loop.__call__ = previous_call
        if previous_marker is None:
            if hasattr(loop, '_h3opt_patched'):
                delattr(loop, '_h3opt_patched')
        else:
            loop._h3opt_patched = previous_marker
    return {'device':str(device), 'tables_bitwise_equal':True,
            'block_stack_comparisons':comparisons, 'outputs_bitwise_equal':True,
            'offloaded_after_each_forward':True, 'rejection_checks':'pass', 'denoiser_callback_once':True, 'stats':stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=Path, required=True)
    args = parser.parse_args()
    entries=json.loads((args.files/'series.json').read_text())['patches']
    entry=next(e for e in entries if e['source']=='h3_runtime/adaln.py')
    original=load('adaln_original',args.files/'original.py',entry['before_sha256'])
    staged=load('adaln_staged',args.files/'staged.py',entry['after_sha256'])
    assert torch.__version__=='2.10.0+cu130', torch.__version__
    assert torch.cuda.device_count()==2
    uuids=[str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-') for i in range(2)]
    assert len(set(uuids))==2
    verify_gpu_allocation(uuids)
    results=[]
    for i in range(2):
        device=torch.device(f'cuda:{i}')
        p=torch.cuda.get_device_properties(i)
        assert '4090' in p.name and (p.major,p.minor)==(8,9) and p.total_memory>=23*2**30
        with torch.cuda.device(device):
            r=check(original,staged,device)
            print(json.dumps(r),flush=True)
            results.append(r)
            gc.collect()
            torch.cuda.empty_cache()
    print(json.dumps({'status':'PASS','scope':'AdaLN plus actual H3 block stack; no full pipeline',
                      'torch':torch.__version__,'uuid':uuids,'results':results}),flush=True)


if __name__=='__main__':
    main()
