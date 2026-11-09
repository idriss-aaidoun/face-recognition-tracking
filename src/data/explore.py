"""Count images and people using the official filename lists.

Download the small train and test lists, not the image archives.
Count unique image paths for each person and check for repeated lines.
Save the counts as CSV files and a JSON summary in outputs/data_exploration.

Run: python -m src.data.explore
"""

import argparse
import csv
import json
import statistics
import urllib.request
from collections import Counter
from pathlib import Path
from urllib.parse import quote

REPO_ID = "RichardErkhov/VGGFace2"


def download_list(split, root, revision="main"):
    """Download one list, or reuse the local copy if it already exists."""
    path = Path(root) / f"{split}_list.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (f"https://huggingface.co/datasets/{REPO_ID}/resolve/"
               f"{quote(revision, safe='')}/data/{split}_list.txt")
        temporary = path.with_suffix(".txt.part")
        print(f"Downloading filename list: {split}", flush=True)
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    return path


def valid_image_path(name):
    """A list entry should look like n000001/0001_01.jpg."""
    parts = name.split("/")
    if len(parts) != 2:
        return False
    identity, filename = parts
    if not identity.startswith("n") or not identity[1:].isdigit():
        return False
    if "\\" in filename:
        return False
    return Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png"}


def inspect_list(path):
    """Count each valid image path once, even if a line is repeated."""
    counts = Counter()
    seen = set()
    lines = 0
    duplicates = 0
    invalid = 0
    with Path(path).open(encoding="utf-8-sig") as source:
        for line in source:
            name = line.strip()
            if not name:
                continue
            lines += 1
            if not valid_image_path(name):
                invalid += 1
                continue
            if name in seen:
                duplicates += 1
                continue
            seen.add(name)
            counts[name.split("/")[0]] += 1
    if not counts:
        raise ValueError(f"No valid image paths found in {path}")
    values = sorted(counts.values())
    minimum_counts = {}
    for minimum in (30, 50, 100, 200):
        eligible = 0
        for count in values:
            if count >= minimum:
                eligible += 1
        minimum_counts[str(minimum)] = eligible

    report = {
        "nonempty_lines": lines,
        "unique_image_paths": sum(values),
        "duplicate_lines": duplicates,
        "invalid_lines": invalid,
        "identities": len(counts),
        "images_per_identity": {
            "min": min(values),
            "median": statistics.median(values),
            "mean": round(statistics.mean(values), 2),
            "max": max(values),
        },
        "identities_with_at_least": minimum_counts,
    }
    return report, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lists-dir", type=Path, default=Path("data/raw/vggface2/lists"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_exploration"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    identities = {}
    for split in ("train", "test"):
        path = download_list(split, args.lists_dir, args.revision)
        reports[split], counts = inspect_list(path)
        identities[split] = set(counts)
        with (args.output / f"{split}_identity_counts.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["identity", "unique_image_paths"])
            writer.writerows(sorted(counts.items()))
    reports["identity_overlap"] = sorted(identities["train"] & identities["test"])
    reports["source"] = REPO_ID
    reports["revision"] = args.revision
    reports["limitations"] = (
        "Counts describe filename lists only. Image existence, quality and pixel "
        "duplicates have not been checked. Duplicate lines count once per image path.")
    result = json.dumps(reports, indent=2)
    (args.output / "summary.json").write_text(result + "\n", encoding="utf-8")
    print(result)
    print(f"Reports saved in {args.output}")


if __name__ == "__main__":
    main()
