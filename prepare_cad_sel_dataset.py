import argparse
import os
import shutil
from pathlib import Path


MODALITY_MAP = {
    "WLE-Set": "NET-WL",
    "EUS-Set": "NET-EUS",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create the legacy CAD-SEL layout expected by train.py/evaluate.py "
            "from the figshare/PDF layout."
        )
    )
    parser.add_argument(
        "--source",
        default="data",
        help="Root containing Images/, Labels/, and metadata.xlsx from the downloaded dataset.",
    )
    parser.add_argument(
        "--destination",
        default="data",
        help="Root where Images/<split>/NET-WL and Images/<split>/NET-EUS will be created.",
    )
    parser.add_argument(
        "--split-name",
        default="Full",
        help="Name of the generated split folder, for example Full, Internal, or External.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["auto", "hardlink", "copy", "symlink"],
        default="auto",
        help="How to materialize files in the generated layout.",
    )
    parser.add_argument(
        "--center-prefix",
        action="store_true",
        help="Prefix patient folders with the center name to avoid patient-name collisions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination files if they are not already linked to the source.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without writing files.",
    )
    return parser.parse_args()


def materialize_file(src, dst, mode, overwrite=False, dry_run=False):
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return "existing"
        except OSError:
            pass

        if not overwrite:
            raise FileExistsError(
                f"Destination already exists and is not the same file: {dst}"
            )

        if not dry_run:
            dst.unlink()

    if dry_run:
        return "created"

    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "auto":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported link mode: {mode}")

    return "created"


def iter_image_files(source_images):
    for center_dir in sorted(source_images.glob("Center-*")):
        if not center_dir.is_dir():
            continue

        center = center_dir.name
        for source_modality, target_modality in MODALITY_MAP.items():
            modality_dir = center_dir / source_modality
            if not modality_dir.is_dir():
                continue

            for category_dir in sorted(modality_dir.iterdir()):
                if not category_dir.is_dir():
                    continue

                category = category_dir.name
                for patient_dir in sorted(category_dir.iterdir()):
                    if not patient_dir.is_dir():
                        continue

                    patient = patient_dir.name
                    for image_path in sorted(patient_dir.rglob("*")):
                        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                            yield {
                                "center": center,
                                "source_modality": source_modality,
                                "target_modality": target_modality,
                                "category": category,
                                "patient": patient,
                                "patient_dir": patient_dir,
                                "image_path": image_path,
                            }


def prepare_dataset(args):
    source_root = Path(args.source)
    destination_root = Path(args.destination)
    source_images = source_root / "Images"
    source_labels = source_root / "Labels"
    destination_images = destination_root / "Images" / args.split_name
    destination_labels = destination_root / "Labels" / args.split_name

    if not source_images.is_dir():
        raise FileNotFoundError(f"Missing source image directory: {source_images}")
    if not source_labels.is_dir():
        raise FileNotFoundError(f"Missing source label directory: {source_labels}")

    seen_patients = {}
    stats = {
        "images_created": 0,
        "images_existing": 0,
        "labels_created": 0,
        "labels_existing": 0,
        "missing_labels": 0,
    }

    for item in iter_image_files(source_images):
        center = item["center"]
        category = item["category"]
        patient = item["patient"]
        target_modality = item["target_modality"]
        source_modality = item["source_modality"]
        patient_dir = item["patient_dir"]
        image_path = item["image_path"]

        target_patient = f"{center}_{patient}" if args.center_prefix else patient
        patient_key = (target_modality, category, target_patient)
        patient_source = (center, source_modality, category, patient)
        if patient_key in seen_patients and seen_patients[patient_key] != patient_source:
            raise ValueError(
                "Patient folder collision while creating "
                f"{target_modality}/{category}/{target_patient}. "
                "Rerun with --center-prefix to make names unique."
            )
        seen_patients[patient_key] = patient_source

        rel_image = image_path.relative_to(patient_dir)
        source_label = (
            source_labels
            / center
            / source_modality
            / category
            / patient
            / rel_image.with_suffix(".txt")
        )
        if not source_label.exists():
            stats["missing_labels"] += 1
            continue

        target_image = (
            destination_images / target_modality / category / target_patient / rel_image
        )
        target_label = (
            destination_labels
            / target_modality
            / category
            / target_patient
            / rel_image.with_suffix(".txt")
        )

        image_status = materialize_file(
            image_path, target_image, args.link_mode, args.overwrite, args.dry_run
        )
        label_status = materialize_file(
            source_label, target_label, args.link_mode, args.overwrite, args.dry_run
        )
        stats[f"images_{image_status}"] += 1
        stats[f"labels_{label_status}"] += 1

    return stats


def main():
    args = parse_args()
    stats = prepare_dataset(args)

    print(f"Created compatibility split: {args.split_name}")
    print(f"Image target: {Path(args.destination) / 'Images' / args.split_name}")
    print(f"Label target: {Path(args.destination) / 'Labels' / args.split_name}")
    print(f"Mode: {args.link_mode}")
    print(f"Images created: {stats['images_created']}")
    print(f"Images already present: {stats['images_existing']}")
    print(f"Labels created: {stats['labels_created']}")
    print(f"Labels already present: {stats['labels_existing']}")
    print(f"Missing labels: {stats['missing_labels']}")

    if stats["missing_labels"]:
        raise SystemExit("Some images did not have matching YOLO label files.")


if __name__ == "__main__":
    main()
