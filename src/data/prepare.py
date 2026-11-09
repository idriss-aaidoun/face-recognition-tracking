"""Download VGGFace2 image archives and unpack them into identity folders.

By default, download the training images. Use --split test for the smaller
pilot archive, or --include-test for both archives. Downloads use the
Hugging Face cache. This file does not split, resize or analyse images.

Run: python -m src.data.prepare --split test
"""

import argparse
import shutil
import tarfile
from pathlib import Path


REPO_ID = "RichardErkhov/VGGFace2"


def extract_archive(archive, destination):
    """Unpack files, but reject links and paths outside the output folder."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as incoming, target.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            else:
                raise ValueError(f"Unsupported archive member: {member.name}")


def download_vggface2(root, revision="main", include_test=False, split="train"):
    """Download the selected archives, then extract their images."""
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    from huggingface_hub import hf_hub_download
    splits = [split]
    if include_test:
        splits = ["train", "test"]

    for split in splits:
        archive = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
            revision=revision, filename=f"data/vggface2_{split}.tar.gz")
        print(f"Extracting {split} images...", flush=True)
        extract_archive(archive, Path(root) / "vggface2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--split", choices=("train", "test"), default="train",
                        help="Download only this partition (default: train)")
    parser.add_argument("--include-test", action="store_true",
                        help="Also download and extract the official test archive")
    # Keep the previous download command working.
    parser.add_argument("--download", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.include_test and args.split == "test":
        parser.error("Use --split test alone to download only test")
    download_vggface2(args.root, args.revision, args.include_test, args.split)
    print(f"VGGFace2 images ready in {args.root / 'vggface2'}")


if __name__ == "__main__":
    main()
