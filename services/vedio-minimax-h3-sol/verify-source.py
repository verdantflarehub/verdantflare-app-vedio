#!/usr/bin/env python3
import argparse
from pathlib import Path
from sol_common import verify_source

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
args = parser.parse_args()
lock = verify_source(args.source)
print(f"Source verified: {lock['revision']}; {len(lock['source_sha256'])} files")
