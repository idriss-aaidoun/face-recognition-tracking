"""Download and extract VGGFace2. Run: python -m src.data.prepare"""

import argparse
from pathlib import Path


REPO_ID = "RichardErkhov/VGGFace2"


def extract_archive(archive, destination):
    """Extract regular files only; reject links and traversal paths."""
    import shutil
    import tarfile
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


def download_vggface2(root, revision="main", include_test=False):
    from huggingface_hub import hf_hub_download
    for split in (("train", "test") if include_test else ("train",)):
        archive = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
            revision=revision, filename=f"data/vggface2_{split}.tar.gz")
        extract_archive(archive, Path(root) / "vggface2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--include-test", action="store_true",
                        help="Also download and extract the official test archive")
    # Keep the previous download command working.
    parser.add_argument("--download", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    download_vggface2(args.root, args.revision, args.include_test)
    print(f"VGGFace2 images ready in {args.root / 'vggface2'}")


if __name__ == "__main__":
    main()
