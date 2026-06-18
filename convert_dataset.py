#!/usr/bin/env python3
"""
CAD-SEL Dataset Converter
Converts multi-center hierarchical structure into flat WLE/EUS + NETs/Non-Nets layout.
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

CLASS_MAP = {
    "NET_G1":      "NETs",
    "NET_G2":      "NETs",
    "Leiomyoma":   "Non-Nets",
    "Lipoma":      "Non-Nets",
    "NonNeoplasm": "Non-Nets",
}

IMAGING_SETS = {"WLE-Set", "EUS-Set"}

# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("cad_converter")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_records(images_root: Path, labels_root: Path, logger: logging.Logger):
    """
    Walk Images/ and build a list of record dicts:
      {center, imaging_set, source_class, patient_folder, filename,
       image_path, label_path (or None)}
    """
    records = []
    unknown_classes = set()

    for center_dir in sorted(images_root.iterdir()):
        if not center_dir.is_dir():
            continue
        center = center_dir.name

        for set_dir in sorted(center_dir.iterdir()):
            if not set_dir.is_dir():
                continue
            imaging_set = set_dir.name
            if imaging_set not in IMAGING_SETS:
                logger.warning(f"Unexpected imaging set '{imaging_set}' — skipping.")
                continue

            for class_dir in sorted(set_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                source_class = class_dir.name
                if source_class not in CLASS_MAP:
                    unknown_classes.add(source_class)
                    logger.warning(f"Unknown class '{source_class}' in {class_dir} — skipping.")
                    continue

                for patient_dir in sorted(class_dir.iterdir()):
                    if not patient_dir.is_dir():
                        continue
                    patient_folder = patient_dir.name

                    for img_file in sorted(patient_dir.iterdir()):
                        if not img_file.is_file():
                            continue

                        # Mirror path in Labels/
                        rel = img_file.relative_to(images_root)
                        label_candidate = labels_root / rel.parent / (img_file.stem + ".*")
                        # glob for any extension
                        label_matches = list(
                            (labels_root / rel.parent).glob(img_file.stem + ".*")
                        ) if (labels_root / rel.parent).exists() else []
                        label_path = label_matches[0] if label_matches else None

                        records.append({
                            "center":         center,
                            "imaging_set":    imaging_set,
                            "source_class":   source_class,
                            "target_class":   CLASS_MAP[source_class],
                            "patient_folder": patient_folder,
                            "filename":       img_file.name,
                            "image_path":     img_file,
                            "label_path":     label_path,
                        })

    if unknown_classes:
        logger.warning(f"Unknown class folders ignored: {unknown_classes}")

    logger.info(f"Discovered {len(records):,} image files across {images_root}")
    return records


# ── Collision detection & resolution ─────────────────────────────────────────

def resolve_collisions(records: list, logger: logging.Logger) -> list:
    """
    Within each (imaging_set, target_class), check for duplicate
    (patient_folder, filename). Disambiguate by prefixing patient_folder
    with source_class when a collision is detected.
    """
    # Group by destination key
    key_to_records = defaultdict(list)
    for r in records:
        key = (r["imaging_set"], r["target_class"], r["patient_folder"], r["filename"])
        key_to_records[key].append(r)

    collision_count = 0
    for key, group in key_to_records.items():
        if len(group) > 1:
            collision_count += len(group)
            imaging_set, target_class, patient_folder, filename = key
            logger.warning(
                f"Collision: {len(group)} files map to "
                f"{imaging_set}/{target_class}/{patient_folder}/{filename} "
                f"(from classes: {[g['source_class'] for g in group]}). "
                f"Disambiguating with source-class prefix."
            )
            for r in group:
                r["patient_folder"] = f"{r['source_class']}_{r['patient_folder']}"

    if collision_count:
        logger.info(f"Resolved {collision_count} collisions via source-class prefix.")
    else:
        logger.info("No patient-folder collisions detected.")

    return records


# ── Pre-flight validation ─────────────────────────────────────────────────────

def preflight(src_root: Path, logger: logging.Logger):
    images_root = src_root / "Images"
    labels_root = src_root / "Labels"
    metadata_src = src_root / "metadata.xlsx"

    errors = []
    if not src_root.exists():
        errors.append(f"Source root not found: {src_root}")
    if not images_root.exists():
        errors.append(f"Images/ directory not found: {images_root}")
    if not labels_root.exists():
        logger.warning(f"Labels/ directory not found: {labels_root}. Labels will be skipped.")
    if not metadata_src.exists():
        logger.warning(f"metadata.xlsx not found: {metadata_src}. Metadata step will be skipped.")

    # Check at least one expected class exists
    found_classes = set()
    if images_root.exists():
        for p in images_root.rglob("*"):
            if p.is_dir() and p.name in CLASS_MAP:
                found_classes.add(p.name)

    missing_classes = set(CLASS_MAP.keys()) - found_classes
    if missing_classes:
        logger.warning(f"These expected class folders were not found anywhere: {missing_classes}")
    else:
        logger.info(f"All 5 class folders confirmed present: {sorted(found_classes)}")

    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)

    return images_root, labels_root if labels_root.exists() else None, metadata_src if metadata_src.exists() else None


# ── Copy engine ───────────────────────────────────────────────────────────────

def copy_records(records: list, dest_root: Path, dry_run: bool, logger: logging.Logger):
    """Copy image and label files to destination tree."""
    copied_images  = 0
    copied_labels  = 0
    missing_labels = 0
    skipped        = 0

    stats = defaultdict(lambda: defaultdict(int))  # [imaging_set][target_class]

    for r in records:
        img_dest = (
            dest_root
            / r["imaging_set"]
            / r["target_class"]
            / r["patient_folder"]
            / r["filename"]
        )

        if not dry_run:
            img_dest.parent.mkdir(parents=True, exist_ok=True)

        # ── Image ──
        if img_dest.exists():
            logger.debug(f"SKIP (exists): {img_dest}")
            skipped += 1
        else:
            if not dry_run:
                shutil.copy2(r["image_path"], img_dest)
            logger.debug(f"COPY image: {r['image_path']} → {img_dest}")
            copied_images += 1
            stats[r["imaging_set"]][r["target_class"]] += 1

        # ── Label ──
        if r["label_path"] is not None:
            lbl_dest = img_dest.parent / r["label_path"].name
            if not lbl_dest.exists():
                if not dry_run:
                    shutil.copy2(r["label_path"], lbl_dest)
                logger.debug(f"COPY label: {r['label_path']} → {lbl_dest}")
                copied_labels += 1
        else:
            logger.debug(f"MISSING label for: {r['image_path']}")
            missing_labels += 1

    return copied_images, copied_labels, missing_labels, skipped, stats


# ── Post-copy validation ──────────────────────────────────────────────────────

def validate_destination(dest_root: Path, records: list, logger: logging.Logger):
    """Assert every expected destination file is present."""
    logger.info("Running post-copy validation …")
    failures = 0
    for r in records:
        img_dest = (
            dest_root
            / r["imaging_set"]
            / r["target_class"]
            / r["patient_folder"]
            / r["filename"]
        )
        if not img_dest.exists():
            logger.error(f"MISSING in destination: {img_dest}")
            failures += 1

    if failures == 0:
        logger.info("✓ All destination image files verified.")
    else:
        logger.error(f"✗ {failures} files missing from destination.")

    return failures == 0


# ── Metadata conversion ───────────────────────────────────────────────────────

def convert_metadata(metadata_src: Path, dest_root: Path, dry_run: bool, logger: logging.Logger):
    logger.info(f"Converting metadata: {metadata_src}")
    try:
        df = pd.read_excel(metadata_src, engine="openpyxl")
    except Exception as e:
        logger.error(f"Failed to read metadata.xlsx: {e}")
        return False

    src_rows, src_cols = df.shape
    logger.info(f"  Source:  {src_rows} rows × {src_cols} columns")

    csv_path = dest_root / "metadata.csv"
    if not dry_run:
        dest_root.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False, encoding="utf-8")

    # Verify round-trip
    if not dry_run:
        df_check = pd.read_csv(csv_path)
        if df_check.shape == df.shape:
            logger.info(f"✓ metadata.csv verified: {df_check.shape[0]} rows × {df_check.shape[1]} columns")
        else:
            logger.error(
                f"✗ Shape mismatch after conversion: "
                f"source {df.shape} vs csv {df_check.shape}"
            )
            return False

    logger.info(f"  Destination: {csv_path}")
    return True


# ── Summary report ────────────────────────────────────────────────────────────

def print_summary(
    records, copied_images, copied_labels,
    missing_labels, skipped, stats, logger
):
    logger.info("=" * 60)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total source records  : {len(records):,}")
    logger.info(f"  Images copied         : {copied_images:,}")
    logger.info(f"  Labels copied         : {copied_labels:,}")
    logger.info(f"  Missing labels        : {missing_labels:,}")
    logger.info(f"  Skipped (exists)      : {skipped:,}")
    logger.info("")
    logger.info("  Per-set / per-class breakdown:")
    for imaging_set in sorted(stats):
        for target_class in sorted(stats[imaging_set]):
            count = stats[imaging_set][target_class]
            logger.info(f"    {imaging_set:10s} / {target_class:10s} : {count:,} images")
    logger.info("=" * 60)


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CAD-SEL dataset into flat WLE/EUS + NETs/Non-Nets layout."
    )
    parser.add_argument(
        "--src", required=True,
        help="Path to the CAD-SEL/ source directory"
    )
    parser.add_argument(
        "--dst", required=True,
        help="Path to the output Dataset/ directory (will be created)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulate all steps without writing any files"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_root  = Path(args.src).resolve()
    dest_root = Path(args.dst).resolve()

    log_path  = dest_root / "conversion.log" if not args.dry_run else Path("conversion_dryrun.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_path)

    mode = "[DRY RUN] " if args.dry_run else ""
    logger.info(f"{mode}CAD-SEL → Dataset converter starting")
    logger.info(f"  Source : {src_root}")
    logger.info(f"  Dest   : {dest_root}")

    # 1. Pre-flight
    images_root, labels_root, metadata_src = preflight(src_root, logger)

    # 2. Discover
    records = discover_records(
        images_root,
        labels_root if labels_root else src_root / "Labels",
        logger
    )

    if not records:
        logger.error("No image files found. Check your source path and folder structure.")
        sys.exit(1)

    # 3. Collision resolution
    records = resolve_collisions(records, logger)

    # 4. Copy files
    logger.info(f"{mode}Copying files …")
    copied_images, copied_labels, missing_labels, skipped, stats = copy_records(
        records, dest_root, args.dry_run, logger
    )

    # 5. Post-copy validation
    if not args.dry_run:
        validate_destination(dest_root, records, logger)

    # 6. Metadata conversion
    if metadata_src:
        convert_metadata(metadata_src, dest_root, args.dry_run, logger)
    else:
        logger.warning("Skipping metadata conversion (source file not found).")

    # 7. Summary
    print_summary(records, copied_images, copied_labels, missing_labels, skipped, stats, logger)
    logger.info(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()