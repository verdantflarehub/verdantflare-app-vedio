#!/usr/bin/env python3
"""Download only locked Ref2VA components to a new, dedicated model root."""
import argparse
import hashlib
import os
from pathlib import Path

from sol_common import ROOT, read_json, sha256, verify_models, write_json


def prepare(destination):
    # mkdir is exclusive: never overwrite or convert an existing shared model tree.
    destination.mkdir(parents=True, exist_ok=False)
    from huggingface_hub import hf_hub_download, hf_hub_url, get_hf_file_metadata

    lock = read_json(ROOT / "models.lock.json")
    receipt = {"lock_sha256": sha256(ROOT / "models.lock.json"), "files": {}}
    for partition in ("base", "adapter"):
        component = lock[partition]
        names = component.get("files", [component.get("file")])
        for name in names:
            kwargs = {"repo_id": component["repo_id"], "filename": name,
                      "revision": component["revision"]}
            metadata = get_hf_file_metadata(hf_hub_url(**kwargs), token=os.getenv("HF_TOKEN"))
            if metadata.commit_hash != component["revision"]:
                raise ValueError("server resolved a different model revision")
            path = Path(hf_hub_download(**kwargs, local_dir=destination / partition))
            if metadata.size is None or path.stat().st_size != metadata.size:
                raise ValueError("download size mismatch")
            digest = sha256(path)
            etag = (metadata.etag or "").strip('"')
            if len(etag) == 64:
                verified = digest == etag
            elif len(etag) == 40:
                blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        blob.update(chunk)
                verified = blob.hexdigest() == etag
            else:
                verified = False
            if not verified:
                raise ValueError("download does not match the pinned repository object")
            if partition == "adapter" and digest != component["sha256"]:
                raise ValueError("unexpected Ref2VA adapter checksum")
            receipt["files"][partition + "/" + name] = {"sha256": digest, "bytes": path.stat().st_size}
    write_json(destination / "models.manifest.json", receipt)
    verify_models(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            verify_models(args.output_dir)
        else:
            prepare(args.output_dir)
    except Exception as error:
        # Network exceptions may contain signed URLs or authorization details.
        print(f"Model preparation failed ({type(error).__name__}); no valid model readiness claimed.")
        return 1
    print("Model integrity verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
