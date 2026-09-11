#!/usr/bin/env python3
"""Read-only inventory/hash check against preserved HF download object receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    lock=json.loads(args.lock.read_text());root=args.models;rows={};errors=[];started=time.monotonic()
    for name in lock['base']['files']:
        base=root/'MiniMax-H3-Diffusers'
        if not (base/name).is_file() and (root/name).is_file():base=root
        file=base/name;meta=base/'.cache/huggingface/download'/(name+'.metadata')
        if not file.is_file() or not meta.is_file():errors.append({'file':name,'error':'missing_file_or_download_receipt'});continue
        lines=meta.read_text().splitlines()
        if len(lines)<2 or lines[0]!=lock['base']['revision']:errors.append({'file':name,'error':'revision_mismatch'});continue
        expected=lines[1];size=file.stat().st_size;sha=hashlib.sha256();git=hashlib.sha1(f'blob {size}\0'.encode())
        with file.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):sha.update(b);git.update(b)
        actual=sha.hexdigest() if len(expected)==64 else git.hexdigest()
        if actual!=expected:errors.append({'file':name,'error':'object_hash_mismatch'})
        rows[name]={'source_relative':str(file.relative_to(root)),'bytes':size,'sha256':sha.hexdigest(),'object_id':expected,'matches_receipt':actual==expected}
        print(json.dumps({'file':name,'verified':actual==expected,'bytes':size}),flush=True)
        if name.endswith('.safetensors.index.json'):
            index=json.loads(file.read_text())
            for shard in set(index['weight_map'].values()):
                path=file.parent/shard
                if not path.resolve().is_relative_to(base.resolve()) or not path.is_file():errors.append({'file':name,'error':'missing_or_escaping_shard'})
    adapter=root/'MiniMax-H3-Turbo'/lock['adapter']['file']
    with adapter.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    if sha!=lock['adapter']['sha256']:errors.append({'file':'adapter','error':'hash_mismatch'})
    result={'status':'PASS' if not errors else 'BLOCKED','source_evidence':'preserved HF revision/object receipts, rehashed now; upstream API currently inaccessible','files':rows,'adapter_sha256':sha,'errors':errors,'wall_s':time.monotonic()-started,'bytes_hashed':sum(x['bytes'] for x in rows.values())+adapter.stat().st_size}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
    return 0 if not errors else 2


if __name__=='__main__':raise SystemExit(main())
