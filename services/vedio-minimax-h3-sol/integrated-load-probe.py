#!/usr/bin/env python3
"""Load-test the experimental full pipeline without submitting a video request."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from gpu_guard import verify_gpu_allocation
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--integrity',type=Path,required=True);args=p.parse_args()
    report=json.loads(args.integrity.read_text())
    # A missing non-runtime FAQ is recorded, never filled with invented content.
    # This probe tests loading only; it does not mark the 64-file model package ready.
    for error in report['errors']:
        if error!={'file':'docs/QA-about-License.md','error':'missing_file_or_download_receipt'}:
            raise ValueError('runtime component integrity failed')
    if len(report['files'])!=64-len(report['errors']) or not all(x['matches_receipt'] for x in report['files'].values()):
        raise ValueError('expected all 61 components plus license and README verified')
    if report['adapter_sha256']!='9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99':raise ValueError('wrong adapter')
    for file in report['files'].values():
        actual=args.models/file['source_relative']
        if not actual.resolve().is_relative_to(args.models.resolve()) or actual.stat().st_size!=file['bytes']:raise ValueError('verified source moved or changed size')
    assert torch.cuda.device_count()==2
    uuids=[str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-') for i in range(2)]
    verify_gpu_allocation(uuids)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    started=time.monotonic()
    from h3_runtime import MiniMaxH3Inference
    print(json.dumps({'stage':'full_pipeline_load','rank':rank,'state':'start'}),flush=True)
    engine=MiniMaxH3Inference(model_path=str(args.models/'MiniMax-H3-Diffusers'),
                            adapter_path=args.models/'MiniMax-H3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors',
                            task='ref2va',attention_backend='dense',cpu_offload=True)
    torch.cuda.synchronize(rank)
    print(json.dumps({'stage':'full_pipeline_load','rank':rank,'state':'passed','wall_s':time.monotonic()-started,
                      'cuda_peak_bytes':torch.cuda.max_memory_allocated(rank),
                      'pipeline_device':str(engine.pipe._execution_device),
                      'transformer_cpu':all(p.device.type=='cpu' for p in engine.transformer.parameters()),
                      'sample_generated':False,'auxiliary_doc_pending':bool(report['errors'])}),flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__=='__main__':main()
