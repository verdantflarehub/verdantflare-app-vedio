#!/usr/bin/env python3
"""Experimental full-pipeline sample entry; requires frozen input and complete runtime integrity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from gpu_guard import verify_gpu_allocation
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--integrity',type=Path,required=True);p.add_argument('--output-width',type=int,choices=[768,1344],required=True);p.add_argument('--output-height',type=int,choices=[768,1344],required=True);p.add_argument('--request',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=json.loads(args.integrity.read_text())
    from sol_common import validate_request, sha256
    request, refs = validate_request(args.request, args.inputs)
    allowed = {'file':'docs/QA-about-License.md','error':'missing_file_or_download_receipt'}
    if (any(e != allowed for e in report['errors']) or len(report['errors']) > 1
            or len(report['files']) != 64-len(report['errors'])
            or not all(x['matches_receipt'] for x in report['files'].values())):
        raise ValueError('complete runtime, LICENSE and README integrity required')
    if report['adapter_sha256']!='9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99':raise ValueError('wrong adapter')
    for file in report['files'].values():
        actual=args.models/file['source_relative']
        if not actual.resolve().is_relative_to(args.models.resolve()) or actual.stat().st_size!=file['bytes']:raise ValueError('verified source moved or changed size')
    assert torch.cuda.device_count()==2
    uuids=[str(torch.cuda.get_device_properties(i).uuid).removeprefix('GPU-') for i in range(2)]
    verify_gpu_allocation(uuids)
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    from retain_cpu_weights import install, verify
    install();verify(torch.device('cuda',rank))
    print(json.dumps({'stage':'retained_cpu_weights','rank':rank,'state':'passed'}),flush=True)
    started=time.monotonic()
    from h3_runtime import MiniMaxH3Inference
    print(json.dumps({'stage':'full_pipeline_load','rank':rank,'state':'start'}),flush=True)
    engine=MiniMaxH3Inference(model_path=str(args.models/'MiniMax-H3-Diffusers'),
                            adapter_path=args.models/'MiniMax-H3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors',
                            task='ref2va',attention_backend='dense',cpu_offload=True,
                            output_width=args.output_width,output_height=args.output_height)
    torch.cuda.synchronize(rank)
    print(json.dumps({'stage':'full_pipeline_load','rank':rank,'state':'passed','wall_s':time.monotonic()-started,
                      'cuda_peak_bytes':torch.cuda.max_memory_allocated(rank),
                      'pipeline_device':str(engine.pipe._execution_device),
                      'transformer_cpu':all(p.device.type=='cpu' for p in engine.transformer.parameters()),
                      'sample_generated':False,'auxiliary_doc_pending':bool(report['errors'])}),flush=True)
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
    references = [MiniMaxH3Reference(**{kind: str(path)}) for kind, path in refs]
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'request.json').write_text(args.request.read_text())
        (args.output/'manifest.json').write_text(json.dumps({
            'status':'running', 'request_sha256':sha256(args.request),
            'quality_review':'pending', 'model_package_complete':not bool(report['errors']),
            'auxiliary_document_errors':report['errors'], 'warmups':1, 'measured_runs':1,
            'engine':'official-sol-h3-with-sealed-cpu-offload-patches',
            'gpu_uuids':uuids, 'width':args.output_width, 'height':args.output_height}, indent=2))
    torch.distributed.barrier()
    print(json.dumps({'stage':'warmup','rank':rank,'state':'start'}),flush=True)
    engine.warmup(duration=request['duration'], prompt=request['prompt'], references=references)
    torch.cuda.reset_peak_memory_stats(rank)
    print(json.dumps({'stage':'generation','rank':rank,'state':'start'}),flush=True)
    result = engine.generate(request['prompt'], duration=request['duration'],
                             seed=request['seed'], references=references)
    if result is not None:
        result.save(args.output/'output.mp4')
        (args.output/'generation.json').write_text(json.dumps({
            'inference_s':result.elapsed_s, 'output_sha256':sha256(args.output/'output.mp4'),
            'media_validation':'pending', 'quality_review':'pending'}, indent=2))
    print(json.dumps({'stage':'generation','rank':rank,'state':'finished',
                      'cuda_peak_bytes':torch.cuda.max_memory_allocated(rank)}),flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__=='__main__':main()
