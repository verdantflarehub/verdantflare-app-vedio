#!/usr/bin/env python3
"""Create a separate, sealed experimental tree; never edit the official source tree."""
import argparse
from pathlib import Path
import shutil
import subprocess

from sol_common import ROOT, read_json, relative_file, sha256, verify_source, write_json


def prepare(source, output):
    source = source.resolve()
    output = output.resolve()
    lock = verify_source(source)
    series = read_json(ROOT / 'patches/series.json')
    if series['upstream_revision'] != lock['revision']:
        raise ValueError('patch series belongs to another upstream revision')
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('experimental tree must be separate from upstream')
    for entry in series['patches']:
        patch = relative_file(ROOT / 'patches', entry['file'])
        if sha256(patch) != entry['sha256']:
            raise ValueError('patch checksum mismatch')
    # Exclusive output and no readiness receipt until every patch verifies.
    output.mkdir(parents=True, exist_ok=False)
    for name in lock['source_sha256']:
        src = relative_file(source, name)
        dst = output / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for entry in series['patches']:
        target = relative_file(output, entry['source'])
        if sha256(target) != entry['before_sha256']:
            raise ValueError('patch input hash mismatch')
        patch = relative_file(ROOT / 'patches', entry['file'])
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)],
                       cwd=output, check=True, capture_output=True)
        if sha256(target) != entry['after_sha256']:
            raise ValueError('patch result hash mismatch')
    expected = dict(lock['source_sha256'])
    for entry in series['patches']:
        expected[entry['source']] = entry['after_sha256']
    for name, digest in expected.items():
        if sha256(relative_file(output, name)) != digest:
            raise ValueError('unexpected change outside declared patch targets')
    actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    if actual != set(expected):
        raise ValueError('unexpected files in patched tree')
    write_json(output / 'patches.manifest.json', {
        'upstream_revision': lock['revision'],
        'series_sha256': sha256(ROOT / 'patches/series.json'),
        'files': expected,
        'status': 'experimental-component-only',
        'inference_enabled': False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.source, args.output)
    except Exception as error:
        print(f'Patched source preparation failed ({type(error).__name__}).')
        return 1
    print('Experimental source sealed; the inference runner remains blocked.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
