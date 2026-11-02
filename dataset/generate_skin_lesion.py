"""
Generate a patient-partitioned skin-lesion dataset for AFL-Lib.

Expected source structure:
    source_root/
        metadata.csv
        images/
            imgs_part_1/
            imgs_part_2/
            imgs_part_3/

Output:
    output_dir/
        config.yaml
        train/0.npz, train/1.npz, ...
        test/0.npz, test/1.npz, ...
        client_mapping.json
        dataset_report.json

Each client corresponds to exactly one patient_id.
Each NPZ stores {'x': NCHW float32 images, 'y': int64 labels}.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from utils.dataset_utils import save_file


CLASS_TO_ID = {
    "BCC": 0,
    "SCC": 1,
    "ACK": 2,
    "SEK": 3,
    "MEL": 4,
    "NEV": 5,
}


def normalize_image_id(value):
    """Normalize img_id / filename to a basename without extension."""
    return Path(str(value).strip()).stem


def build_image_index(images_root):
    """Index images recursively by filename stem."""
    index = defaultdict(list)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    for path in Path(images_root).rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            index[path.stem].append(path)

    return index


def load_image(path, image_size):
    """Return normalized RGB image in CHW layout, float32 in [-1, 1]."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0

    array = np.transpose(array, (2, 0, 1))
    array = (array - 0.5) / 0.5
    return array.astype(np.float32)


def split_patient_samples(samples, train_ratio, rng):
    """
    Split one patient's samples. For patients with >=2 images, ensure
    at least one image in train and one in test. A one-image patient
    is kept in train because a leakage-free within-patient split is
    impossible.
    """
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    if len(indices) == 1:
        train_idx = indices
        test_idx = np.array([], dtype=int)
    else:
        n_train = int(round(len(indices) * train_ratio))
        n_train = min(max(n_train, 1), len(indices) - 1)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in test_idx]
    return train, test


def generate_dataset(source_root, output_dir, image_size=224,
                     train_ratio=0.75, seed=1):
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    metadata_path = source_root / "metadata.csv"
    images_root = source_root / "images"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    image_index = build_image_index(images_root)
    if not image_index:
        raise RuntimeError(f"No supported image files found under {images_root}")

    grouped = defaultdict(list)
    missing_images = []
    unknown_labels = []
    duplicate_ids = []
    used_image_paths = set()

    with metadata_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"patient_id", "diagnostic", "img_id"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"metadata.csv is missing columns: {sorted(missing_columns)}")

        for row_number, row in enumerate(reader, start=2):
            patient_id = str(row.get("patient_id", "")).strip()
            image_id = normalize_image_id(row.get("img_id", ""))
            diagnostic = str(row.get("diagnostic", "")).strip().upper()

            if not patient_id or patient_id.lower() == "nan":
                print(f"WARNING: row {row_number}: missing patient_id; skipping")
                continue
            if not image_id or image_id.lower() == "nan":
                print(f"WARNING: row {row_number}: missing img_id; skipping")
                continue
            if diagnostic not in CLASS_TO_ID:
                unknown_labels.append({
                    "row": row_number,
                    "img_id": image_id,
                    "diagnostic": diagnostic
                })
                continue

            matches = image_index.get(image_id, [])
            if not matches:
                missing_images.append(image_id)
                continue
            if len(matches) > 1:
                duplicate_ids.append({
                    "img_id": image_id,
                    "paths": [str(p) for p in matches]
                })
                # Do not silently choose between ambiguous files.
                continue

            image_path = matches[0]
            if image_path in used_image_paths:
                print(f"WARNING: image referenced more than once; skipping: {image_path}")
                continue

            used_image_paths.add(image_path)
            grouped[patient_id].append({
                "path": image_path,
                "label": CLASS_TO_ID[diagnostic],
                "img_id": image_id,
            })

    if not grouped:
        raise RuntimeError(
            "No usable samples found. Check metadata img_id values, "
            "diagnostic labels, and image filenames."
        )

    patient_ids = sorted(grouped)
    train_data = []
    test_data = []
    client_mapping = {}
    report_clients = []
    total_train = 0
    total_test = 0

    for client_id, patient_id in enumerate(patient_ids):
        patient_samples = grouped[patient_id]
        # Deterministic order before seeded split.
        patient_samples.sort(key=lambda item: item["img_id"])

        train_samples, test_samples = split_patient_samples(
            patient_samples, train_ratio, rng
        )

        def materialize(samples):
            if not samples:
                return (
                    np.empty((0, 3, image_size, image_size), dtype=np.float32),
                    np.empty((0,), dtype=np.int64)
                )
            images = np.stack([
                load_image(sample["path"], image_size) for sample in samples
            ])
            labels = np.asarray(
                [sample["label"] for sample in samples], dtype=np.int64
            )
            return images, labels

        x_train, y_train = materialize(train_samples)
        x_test, y_test = materialize(test_samples)

        train_data.append({"x": x_train, "y": y_train})
        test_data.append({"x": x_test, "y": y_test})

        train_counts = Counter(int(v) for v in y_train)
        test_counts = Counter(int(v) for v in y_test)
        all_counts = Counter(
            int(sample["label"]) for sample in patient_samples
        )

        client_mapping[str(client_id)] = {
            "patient_id": patient_id,
            "num_images": len(patient_samples),
        }
        report_clients.append({
            "client_id": client_id,
            "patient_id": patient_id,
            "num_images": len(patient_samples),
            "train_images": len(train_samples),
            "test_images": len(test_samples),
            "class_counts": {
                name: all_counts[class_id]
                for name, class_id in CLASS_TO_ID.items()
                if all_counts[class_id] > 0
            },
            "train_class_counts": {
                name: train_counts[class_id]
                for name, class_id in CLASS_TO_ID.items()
                if train_counts[class_id] > 0
            },
            "test_class_counts": {
                name: test_counts[class_id]
                for name, class_id in CLASS_TO_ID.items()
                if test_counts[class_id] > 0
            },
        })

        total_train += len(train_samples)
        total_test += len(test_samples)

        print(
            f"Client {client_id:4d} | patient={patient_id} | "
            f"total={len(patient_samples):4d} | "
            f"train={len(train_samples):4d} | test={len(test_samples):4d} | "
            f"classes={dict(all_counts)}"
        )

    # Config keys retain the framework's naming conventions.
    cfg = {
        "dataset": "skin_lesion",
        "dir_path": str(output_dir),
        "partition": "patient",
        "client_num": len(patient_ids),
        "class_num": len(CLASS_TO_ID),
        "train_ratio": float(train_ratio),
        "batch_size": 10,
        "image_size": int(image_size),
        "image_channels": 3,
        "seed": int(seed),
        "class_to_id": CLASS_TO_ID,
        "source_metadata": str(metadata_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    # The framework's save_file expects these directories to exist.
    (output_dir / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "test").mkdir(parents=True, exist_ok=True)

    # Remove stale client files from an earlier run with more clients.
    for split in ("train", "test"):
        for old_file in (output_dir / split).glob("*.npz"):
            old_file.unlink()

    save_file(train_data, test_data, cfg)

    with (output_dir / "client_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(client_mapping, f, indent=2)

    report = {
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "class_to_id": CLASS_TO_ID,
        "client_num": len(patient_ids),
        "total_images": total_train + total_test,
        "train_images": total_train,
        "test_images": total_test,
        "patients_with_one_image": [
            patient_id for patient_id in patient_ids
            if len(grouped[patient_id]) == 1
        ],
        "missing_images": missing_images,
        "unknown_labels": unknown_labels,
        "ambiguous_image_ids": duplicate_ids,
        "clients": report_clients,
    }
    with (output_dir / "dataset_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n========== Dataset summary ==========")
    print(f"Patients / clients: {len(patient_ids)}")
    print(f"Images: {total_train + total_test}")
    print(f"Training images: {total_train}")
    print(f"Testing images: {total_test}")
    print(f"Missing image files: {len(missing_images)}")
    print(f"Unknown diagnostic labels: {len(unknown_labels)}")
    print(f"Ambiguous image IDs: {len(duplicate_ids)}")
    print(f"Output: {output_dir.resolve()}")
    print("=====================================")

    if duplicate_ids:
        print("WARNING: Ambiguous img_id values were skipped. See dataset_report.json.")
    if missing_images:
        print("WARNING: Some metadata images were not found. See dataset_report.json.")
    if unknown_labels:
        print("WARNING: Some rows had unsupported labels. See dataset_report.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a patient-partitioned AFL-Lib skin-lesion dataset."
    )
    parser.add_argument(
        "--source-root", required=True,
        help="Directory containing metadata.csv and images/."
    )
    parser.add_argument(
        "--output-dir", default="skin_lesion",
        help="AFL-Lib dataset output directory (default: skin_lesion)."
    )
    parser.add_argument(
        "--image-size", type=int, default=224,
        help="Square image resolution (default: 224)."
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.75,
        help="Per-patient training fraction (default: 0.75)."
    )
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Random seed (default: 1)."
    )
    args = parser.parse_args()

    generate_dataset(
        source_root=args.source_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
