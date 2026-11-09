"""Look at the images already downloaded on this computer.

Count images and people, then check a random sample for image sizes,
unreadable files and exact duplicates. Save CSV tables, a JSON summary,
charts and face grids in outputs/image_exploration. Source images stay unchanged.

Run: python -m src.data.explore_images
"""

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, ImageStat


def describe(values):
    """Return four basic statistics for a list of numbers."""
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 2),
        "max": max(values),
    }


def histogram(values, title, output, bins=20):
    """Group numbers into bins and draw a simple bar chart."""
    canvas = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 15), title, fill="black")
    low, high = min(values), max(values)
    width = (high - low + 1) / bins
    counts = Counter()
    for value in values:
        bin_number = int((value - low) / width)
        bin_number = min(bins - 1, bin_number)
        counts[bin_number] += 1
    peak = max(counts.values())
    for i in range(bins):
        x = 65 + i * 39
        height = int(300 * counts[i] / peak)
        if height:
            draw.rectangle((x, 350 - height, x + 35, 350), fill="#396ab1")
    draw.line((60, 45, 60, 350, 850, 350), fill="black", width=2)
    draw.text((10, 45), str(peak), fill="black")
    draw.text((60, 365), str(low), fill="black")
    draw.text((800, 365), str(high), fill="black")
    draw.text((60, 390), "Horizontal: value | Vertical: number of observations", fill="black")
    canvas.save(output)


def gallery(paths, output, columns=6):
    """Save up to 24 small images, labelled with their identity folder."""
    paths = paths[:24]
    row_count = (len(paths) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 150, row_count * 175), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths):
        x = (index % columns) * 150
        y = (index // columns) * 175
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((140, 140))
            canvas.paste(image, (x + (150 - image.width) // 2, y))
            draw.text((x + 4, y + 143), path.parent.name, fill="black")
        except (OSError, ValueError):
            draw.text((x + 4, y + 50), "Unreadable", fill="red")
    canvas.save(output)


def find_images(root):
    """Read image paths from identity folders and count images per person."""
    paths = []
    for path in root.glob("n*/*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            paths.append(path)
    paths.sort()
    if not paths:
        raise ValueError("No images found")

    counts = Counter()
    for path in paths:
        counts[path.parent.name] += 1
    return paths, counts


def check_completeness(paths, root, list_path):
    """Compare local paths with the official list, if that list is available."""
    if list_path is None or not Path(list_path).is_file():
        return {"checked": False}

    actual = set()
    for path in paths:
        actual.add(path.relative_to(root).as_posix())
    expected = set(Path(list_path).read_text(encoding="utf-8-sig").split())
    missing = expected - actual
    extra = actual - expected
    return {
        "checked": True,
        "expected_paths": len(expected),
        "missing_paths": len(missing),
        "unexpected_paths": len(extra),
        "complete": not missing and not extra,
    }


def inspect_sample(sample, root):
    """Decode sampled images and record sizes, brightness and duplicate pairs."""
    rows = []
    errors = []
    hashes = {}
    duplicates = []
    for path in sample:
        try:
            with Image.open(path) as source:
                source.load()
                mode = source.mode
                image = ImageOps.exif_transpose(source).convert("RGB")
            relative_path = path.relative_to(root).as_posix()

            # The same RGB pixels and size mean an exact duplicate.
            pixel_data = str(image.size).encode() + image.tobytes()
            digest = hashlib.sha256(pixel_data).hexdigest()
            if digest in hashes:
                duplicates.append([hashes[digest], relative_path])
            else:
                hashes[digest] = relative_path

            gray = image.convert("L").resize((64, 64))
            brightness = ImageStat.Stat(gray).mean[0]
            rows.append({
                "path": relative_path,
                "identity": path.parent.name,
                "width": image.width,
                "height": image.height,
                "original_mode": mode,
                "aspect_ratio": round(image.width / image.height, 3),
                "brightness": round(brightness, 2),
            })
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            errors.append({"path": path.relative_to(root).as_posix(), "error": str(error)})
    if not rows:
        raise ValueError("No sampled images could be decoded")
    return rows, errors, duplicates


def save_tables(output, counts, rows):
    """Save the full identity counts and the sampled image measurements."""
    with (output / "identity_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["identity", "images"])
        writer.writerows(sorted(counts.items()))
    with (output / "sample_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def explore(root="data/raw/vggface2/test", output="outputs/image_exploration",
            sample_size=1000, seed=42, list_path="data/raw/vggface2/lists/test_list.txt"):
    """Run the inventory, sample checks, and report generation in order."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    root = Path(root)
    output = Path(output)
    if not root.is_dir():
        raise FileNotFoundError(root)
    output.mkdir(parents=True, exist_ok=True)
    paths, counts = find_images(root)
    completeness = check_completeness(paths, root, list_path)
    rng = random.Random(seed)
    sample = rng.sample(paths, min(sample_size, len(paths)))
    print(f"Found {len(paths)} images and {len(counts)} identities.", flush=True)
    print(f"Checking a sample of {len(sample)} images...", flush=True)
    rows, errors, duplicates = inspect_sample(sample, root)

    save_tables(output, counts, rows)

    modes = Counter()
    small_images = 0
    for row in rows:
        modes[row["original_mode"]] += 1
        if row["width"] < 128 or row["height"] < 128:
            small_images += 1

    report = {
        "root": str(root),
        "seed": seed,
        "images_on_disk": len(paths),
        "identities_on_disk": len(counts),
        "completeness": completeness,
        "images_per_identity": describe(list(counts.values())),
        "sample_requested": sample_size,
        "sample_checked": len(sample),
        "sample_readable": len(rows),
        "sample_errors": errors,
        "sample_original_modes": dict(modes),
        "sample_width": describe([row["width"] for row in rows]),
        "sample_height": describe([row["height"] for row in rows]),
        "sample_smaller_than_128": small_images,
        "sample_exact_pixel_duplicate_pairs": duplicates,
        "limitations": (
            "Inventory covers images on disk at scan time. Quality and exact pixel "
            "duplicate checks cover only the random sample. Near duplicates, label "
            "correctness, demographic balance and recording sessions are not verified. "
            "No splits or image changes were made. Official test identities used for "
            "development cannot serve as untouched final unseen identities."
        ),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    histogram(list(counts.values()), "Images per identity - all local identities", output / "identity_counts.png")
    histogram([row["width"] for row in rows], "Image widths - random sample", output / "widths.png")
    histogram([row["height"] for row in rows], "Image heights - random sample", output / "heights.png")
    histogram([row["brightness"] for row in rows], "Mean brightness (0-255) - random sample", output / "brightness.png")
    gallery(sample, output / "random_faces.png")
    # Six people, four images each: inspect variation within an identity.
    identities = rng.sample(sorted(counts), min(6, len(counts)))
    grouped = []
    for identity in identities:
        candidates = [path for path in paths if path.parent.name == identity]
        grouped.extend(rng.sample(candidates, min(4, len(candidates))))
    gallery(grouped, output / "same_identity_faces.png", columns=4)
    print(json.dumps(report, indent=2))
    print(f"Reports and figures saved in {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/vggface2/test")
    parser.add_argument("--output", default="outputs/image_exploration")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-path", default="data/raw/vggface2/lists/test_list.txt")
    args = parser.parse_args()
    explore(args.root, args.output, args.sample_size, args.seed, args.list_path)


if __name__ == "__main__":
    main()
