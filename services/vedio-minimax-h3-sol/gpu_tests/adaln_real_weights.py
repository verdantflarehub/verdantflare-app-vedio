"""Real H3 timestep modules and two AdaLN projections, mounted read-only."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path

from gpu_guard import verify_gpu_allocation
import torch
from safetensors import safe_open
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from adaln_offload import load


def real_modules(root):
    model=root/'MiniMax-H3-Diffusers/transformer_ref'
    index=json.loads((model/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    config=json.loads((model/'config.json').read_text())
    with torch.device('meta'):
        full=MiniMaxH3Transformer3DModel.from_config(config)
    stack=torch.nn.Module()
    stack.time_proj=full.time_proj
    stack.time_embedder=full.time_embedder
    stack.transformer_blocks=torch.nn.ModuleList()
    for i in (0,49):
        block=torch.nn.Module()
        block.adaln_proj=full.transformer_blocks[i].adaln_proj
        stack.transformer_blocks.append(block)
    mapping={name:name for name in stack.state_dict()}
    mapping.update({name:name.replace('transformer_blocks.1.','transformer_blocks.49.')
                    for name in mapping if name.startswith('transformer_blocks.1.')})
    tensors={}
    hashes={}
    for shard in sorted({index[name] for name in mapping.values()}):
        file=model/shard
        assert file.resolve().is_relative_to(model.resolve())
        with safe_open(file,framework='pt',device='cpu') as source:
            for target,name in mapping.items():
                if index[name]!=shard:continue
                tensor=source.get_tensor(name)
                tensors[target]=tensor
                hashes[name]={'dtype':str(tensor.dtype),'shape':list(tensor.shape),
                              'sha256':hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
    stack.load_state_dict(tensors,strict=True,assign=True)
    assert all(p.device.type=='cpu' for p in stack.parameters())
    return stack.eval(), hashes


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--files',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    args=parser.parse_args()
    entry=next(e for e in json.loads((args.files/'series.json').read_text())['patches'] if e['source']=='h3_runtime/adaln.py')
    original=load('real_adaln_original',args.files/'original.py',entry['before_sha256'])
    staged=load('real_adaln_staged',args.files/'staged.py',entry['after_sha256'])
    assert torch.__version__=='2.10.0+cu130'
    assert torch.cuda.device_count()==2
    uuids=[str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-') for i in range(2)]
    verify_gpu_allocation(uuids)
    source,hashes=real_modules(args.models)
    # Use the official four-step video/audio scheduler shifts on the pinned scheduler configs.
    from diffusers import MiniMaxH3Scheduler
    model=args.models/'MiniMax-H3-Diffusers'
    video_scheduler=MiniMaxH3Scheduler.from_pretrained(model,subfolder='scheduler',local_files_only=True)
    audio_scheduler=MiniMaxH3Scheduler.from_pretrained(model,subfolder='audio_scheduler',local_files_only=True)
    video_scheduler.set_shift(12.)
    audio_scheduler.set_shift(3.)
    # Component comparison uses both schedules; this does not submit a video request.
    video_scheduler.set_timesteps(5)
    audio_scheduler.set_timesteps(5)
    video=video_scheduler.timesteps
    audio=audio_scheduler.timesteps
    results=[]
    for i in range(2):
        device=torch.device(f'cuda:{i}')
        prop=torch.cuda.get_device_properties(i)
        assert '4090' in prop.name and (prop.major,prop.minor)==(8,9) and prop.total_memory>=23*2**30
        with torch.cuda.device(device):
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
            resident=copy.deepcopy(source).to(device)
            baseline=original.precompute(resident,video,audio,include_audio_condition=True)
            torch.cuda.synchronize(device)
            resident_peak=torch.cuda.max_memory_allocated(device)
            expected=[b.adaln_proj.table.cpu() for b in resident.transformer_blocks]
            del resident
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
            candidate=copy.deepcopy(source)
            stats=staged.precompute(candidate,video,audio,include_audio_condition=True,staging_device=device)
            torch.cuda.synchronize(device)
            staged_peak=torch.cuda.max_memory_allocated(device)
            assert stats==baseline
            for table,block in zip(expected,candidate.transformer_blocks):assert torch.equal(table,block.adaln_proj.table)
            assert all(p.device.type=='cpu' for p in candidate.parameters())
            assert staged_peak<resident_peak
            result={'device':str(device),'tables_bitwise_equal':True,'resident_peak_bytes':resident_peak,
                    'staged_peak_bytes':staged_peak,'stats':stats}
            results.append(result);print(json.dumps(result),flush=True)
            del candidate,expected
    print(json.dumps({'status':'PASS','scope':'real AdaLN slices; not full model readiness',
                      'torch':torch.__version__,'uuid':uuids,'source_tensors':hashes,
                      'video_timesteps':video.tolist(),'audio_timesteps':audio.tolist(),'results':results}),flush=True)


if __name__=='__main__':main()
