#!/usr/bin/env python3
"""
CAD-SEL Preprocessing Script
----------------------------
Performs patient-level 80/20 train/test split on the converted CAD-SEL dataset,
following the methodology described in:

  Xu, L., Chen, S., Li, C. et al. "CAD-SEL: A dual-modal colonoscopy dataset
  of subepithelial lesion." Scientific Data (2026).
  https://doi.org/10.1038/s41597-026-07331-y

Key design choices (per the paper):
  - Patient-level splitting (NOT image-level) to prevent data leakage
  - Stratified sampling to preserve class distributions across splits
  - WLE and EUS images from the same patient always go to the same split
  - Reproducible split via a fixed random seed with split-CSV output
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent  # d:\s7\CAD-SEL

DEFAULT_SRC_DIR = REPO_ROOT / "data" / "CAD-SEL-Dataset-New"

# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, log_filename: str = "preprocess.log") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cad_preprocess")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_dir / log_filename, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ── Patient discovery ────────────────────────────────────────────────────────

# Map patient folder prefix → fine-grained category
PREFIX_TO_CATEGORY = {
    "NET_G1":      "NET_G1",
    "NET_G2":      "NET_G2",
    "Leiomyoma":   "Leiomyoma",
    "Lipoma":      "Lipoma",
    "NonNeoplasm": "NonNeoplasm",
}

# Map prefix → binary class (per the paper: NET = malignant/potentially malignant)
PREFIX_TO_BINARY = {
    "NET_G1":      "NET",
    "NET_G2":      "NET",
    "Leiomyoma":   "non-NET",
    "Lipoma":      "non-NET",
    "NonNeoplasm": "non-NET",
}

IMAGING_SETS = ["WLE-Set", "EUS-Set"]
TARGET_CLASSES = ["NETs", "Non-Nets"]


def discover_patients(src_dir: Path, logger: logging.Logger) -> dict:
    """
    Walk the flat dataset and build a patient-centric dictionary.

    Returns
    -------
    dict[str, dict]
        Keyed by patient_name (e.g. "NET_G1-0"), value:
          {
            "patient_name": str,
            "prefix":       str,   # e.g. "NET_G1"
            "category":     str,   # fine-grained category
            "binary":       str,   # "NET" or "non-NET"
            "modalities": {
                "WLE-Set": {"NETs"|"Non-Nets": Path, ...},
                "EUS-Set": {"NETs"|"Non-Nets": Path, ...}
            },
            "num_wle_images": int,
            "num_eus_images": int,
          }
    """
    patients: dict[str, dict] = {}

    for imaging_set in IMAGING_SETS:
        set_path = src_dir / imaging_set
        if not set_path.is_dir():
            logger.warning(f"Imaging set not found: {set_path} — skipping.")
            continue

        for target_class in TARGET_CLASSES:
            class_path = set_path / target_class
            if not class_path.is_dir():
                logger.warning(f"Class folder not found: {class_path} — skipping.")
                continue

            for patient_dir in sorted(class_path.iterdir()):
                if not patient_dir.is_dir():
                    continue
                patient_name = patient_dir.name  # e.g. "NET_G1-0"

                # Determine prefix
                prefix = next(
                    (p for p in PREFIX_TO_CATEGORY if patient_name.startswith(p)),
                    None,
                )
                if prefix is None:
                    # Try matching with underscore: "Leiomyoma_Leiomyoma-0" from collision resolution
                    parts = patient_name.split("_", 1)
                    if len(parts) == 2:
                        prefix = next(
                            (p for p in PREFIX_TO_CATEGORY if parts[1].startswith(p) or parts[0].startswith(p)),
                            None,
                        )
                if prefix is None:
                    logger.warning(f"Could not determine prefix for: {patient_name} — skipping.")
                    continue

                if patient_name not in patients:
                    patients[patient_name] = {
                        "patient_name": patient_name,
                        "prefix":       prefix,
                        "category":     PREFIX_TO_CATEGORY[prefix],
                        "binary":       PREFIX_TO_BINARY[prefix],
                        "modalities":   {},
                        "num_wle_images": 0,
                        "num_eus_images": 0,
                    }

                patients[patient_name]["modalities"].setdefault(imaging_set, {})[target_class] = patient_dir

                # Count images
                n_images = sum(1 for f in patient_dir.iterdir()
                               if f.is_file() and f.suffix.lower() in (".tiff", ".tif", ".png", ".jpg", ".jpeg"))
                if imaging_set == "WLE-Set":
                    patients[patient_name]["num_wle_images"] = n_images
                else:
                    patients[patient_name]["num_eus_images"] = n_images

    logger.info(f"Discovered {len(patients)} unique patients across {src_dir}")
    return patients


# ── Split logic ──────────────────────────────────────────────────────────────

def create_patient_split(
    patients: dict,
    test_size: float = 0.20,
    random_seed: int = 42,
    logger: logging.Logger = None,
) -> tuple[list[str], list[str]]:
    """
    Perform a stratified patient-level train/test split.

    Stratification is done on the fine-grained category to preserve
    class distribution. Patients with only a few examples in rare
    classes (e.g. Leiomyoma, 24 patients) are assigned randomly
    within their stratum.

    Returns (train_patients, test_patients) — lists of patient names.
    """
    patient_names = sorted(patients.keys())
    categories = [patients[p]["category"] for p in patient_names]

    # Count patients per category
    cat_counts = pd.Series(categories).value_counts()
    if logger:
        logger.info("Patient distribution by category:")
        for cat, cnt in cat_counts.items():
            logger.info(f"  {cat:15s}: {cnt:4d} patients")

    # For categories with >= 5 patients, use stratified split.
    # For singleton categories (shouldn't happen here), assign to train.
    try:
        train_names, test_names, train_cats, test_cats = train_test_split(
            patient_names,
            categories,
            test_size=test_size,
            random_state=random_seed,
            stratify=categories,
        )
    except ValueError as e:
        # If some categories have too few samples for stratification,
        # fall back to unstratified split.
        if logger:
            logger.warning(f"Stratified split failed ({e}) — falling back to random split.")
        train_names, test_names = train_test_split(
            patient_names,
            test_size=test_size,
            random_state=random_seed,
        )

    if logger:
        logger.info(f"Split: {len(train_names)} train / {len(test_names)} test patients "
                     f"({len(train_names) / len(patient_names):.1%} / {len(test_names) / len(patient_names):.1%})")

        # Report per-category split
        for cat in sorted(cat_counts.index):
            train_c = sum(1 for p in train_names if patients[p]["category"] == cat)
            test_c  = sum(1 for p in test_names  if patients[p]["category"] == cat)
            logger.info(f"  {cat:15s}: {train_c:4d} train / {test_c:4d} test")

    return train_names, test_names


# ── Copy engine ──────────────────────────────────────────────────────────────

def copy_split(
    src_dir: Path,
    dest_dir: Path,
    patients: dict,
    split_patients: list[str],
    split_name: str,
    logger: logging.Logger,
) -> dict:
    """
    Copy patient folders into dest_dir/<split_name>/ maintaining the
    WLE-Set/EUS-Set → NETs/Non-Nets hierarchy.

    Returns stats dict.
    """
    split_root = dest_dir / split_name
    stats = defaultdict(lambda: defaultdict(int))  # [imaging_set][target_class]

    for patient_name in split_patients:
        pinfo = patients[patient_name]
        for imaging_set, class_map in pinfo["modalities"].items():
            for target_class, src_patient_dir in class_map.items():
                dst_patient_dir = split_root / imaging_set / target_class / patient_name
                dst_patient_dir.mkdir(parents=True, exist_ok=True)

                for f in src_patient_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst_patient_dir / f.name)
                        stats[imaging_set][target_class] += 1

    # Log summary
    logger.info(f"  [{split_name}] files copied:")
    total = 0
    for imaging_set in IMAGING_SETS:
        for target_class in TARGET_CLASSES:
            n = stats[imaging_set].get(target_class, 0)
            if n > 0:
                logger.info(f"    {imaging_set:10s} / {target_class:10s}: {n:5d} files")
                total += n
    logger.info(f"    {'TOTAL':>23s}: {total:5d} files")
    return stats


# ── Split CSV generation ─────────────────────────────────────────────────────

def write_split_csv(
    dest_dir: Path,
    patients: dict,
    train_names: list[str],
    test_names: list[str],
    metadata_df: pd.DataFrame or None,
    logger: logging.Logger,
):
    """
    Write train_split.csv and test_split.csv with patient-level
    information merged from metadata.
    """
    for split_name, split_list in [("train", train_names), ("test", test_names)]:
        rows = []
        for pname in split_list:
            p = patients[pname]
            row = {
                "patient_name": pname,
                "prefix":       p["prefix"],
                "category":     p["category"],
                "binary":       p["binary"],
                "num_wle_images": p["num_wle_images"],
                "num_eus_images": p["num_eus_images"],
            }
            # Merge metadata if available
            if metadata_df is not None and "Name" in metadata_df.columns:
                match = metadata_df[metadata_df["Name"] == pname]
                if not match.empty:
                    for col in metadata_df.columns:
                        if col not in row:
                            row[col] = match.iloc[0][col]
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = dest_dir / f"{split_name}_split.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Wrote {csv_path} ({len(df)} patients)")

        # Print distribution
        if logger:
            logger.info(f"  [{split_name}] category distribution:")
            for cat, cnt in df["category"].value_counts().items():
                logger.info(f"    {cat:15s}: {cnt:4d}")


# ── Validation ───────────────────────────────────────────────────────────────

def validate_split(
    src_dir: Path,
    dest_dir: Path,
    patients: dict,
    train_names: list[str],
    test_names: list[str],
    logger: logging.Logger,
) -> bool:
    """Verify no patient overlap and all expected files exist."""
    logger.info("Running split validation …")

    # 1. No overlap
    train_set = set(train_names)
    test_set  = set(test_names)
    overlap = train_set & test_set
    if overlap:
        logger.error(f"Patient overlap detected! {overlap}")
        return False
    logger.info("[OK] No patient overlap between train and test splits.")

    # 2. All patients accounted for
    all_patients = set(patients.keys())
    split_union = train_set | test_set
    if all_patients != split_union:
        missing = all_patients - split_union
        extra   = split_union - all_patients
        if missing:
            logger.error(f"Patients missing from splits: {missing}")
        if extra:
            logger.error(f"Extra patients in splits: {extra}")
        return False
    logger.info("[OK] All patients accounted for in splits.")

    # 3. Spot-check file counts
    for split_name, split_list in [("train", train_names), ("test", test_names)]:
        split_root = dest_dir / split_name
        for pname in split_list:
            p = patients[pname]
            for imaging_set, class_map in p["modalities"].items():
                for target_class in class_map:
                    src_dir_p = class_map[target_class]
                    dst_dir_p = split_root / imaging_set / target_class / pname
                    if not dst_dir_p.is_dir():
                        logger.error(f"Missing: {dst_dir_p}")
                        return False
                    src_files = set(f.name for f in src_dir_p.iterdir() if f.is_file())
                    dst_files = set(f.name for f in dst_dir_p.iterdir() if f.is_file())
                    if src_files != dst_files:
                        logger.error(f"File mismatch in {pname}: src={src_files - dst_files}, dst={dst_files - src_files}")
                        return False
        logger.info(f"[OK] [{split_name}] all {len(split_list)} patient folders verified.")

    return True


# ── Metadata loading ─────────────────────────────────────────────────────────

def load_metadata(src_dir: Path, logger: logging.Logger) -> pd.DataFrame or None:
    csv_path = src_dir / "metadata.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded metadata: {len(df)} rows from {csv_path}")
        return df
    else:
        logger.warning(f"metadata.csv not found at {csv_path} — proceeding without metadata.")
        return None


# ── Main pipeline ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CAD-SEL Preprocessing — patient-level 80/20 train/test split"
    )
    parser.add_argument(
        "--src",
        default=str(DEFAULT_SRC_DIR),
        help=f"Path to the converted CAD-SEL dataset (default: {DEFAULT_SRC_DIR})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion of patients for the test split (default: 0.20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without writing files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src).resolve()
    dest_dir = src_dir  # train/ and test/ go inside the source directory

    if not src_dir.is_dir():
        print(f"ERROR: Source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    logger = setup_logging(dest_dir, "preprocess.log")
    logger.info("=" * 60)
    logger.info("CAD-SEL PREPROCESSING — PATIENT-LEVEL TRAIN/TEST SPLIT")
    logger.info("=" * 60)
    logger.info(f"  Source     : {src_dir}")
    logger.info(f"  Destination: {dest_dir}")
    logger.info(f"  Test size  : {args.test_size:.0%}")
    logger.info(f"  Random seed: {args.seed}")
    logger.info(f"  Dry run    : {args.dry_run}")
    logger.info("")

    # ── Step 1: Discover patients ────────────────────────────────────────
    logger.info("Step 1: Discovering patients …")
    patients = discover_patients(src_dir, logger)

    if len(patients) == 0:
        logger.error("No patients found. Aborting.")
        sys.exit(1)

    # ── Step 2: Load metadata ────────────────────────────────────────────
    logger.info("Step 2: Loading metadata …")
    metadata_df = load_metadata(src_dir, logger)

    # ── Step 3: Create patient-level split ───────────────────────────────
    logger.info("Step 3: Creating patient-level train/test split …")
    train_names, test_names = create_patient_split(
        patients,
        test_size=args.test_size,
        random_seed=args.seed,
        logger=logger,
    )

    if args.dry_run:
        logger.info("Dry run — stopping before file copy.")
        logger.info(f"Would create: {dest_dir / 'train'} and {dest_dir / 'test'}")
        logger.info(f"Train patients: {len(train_names)}")
        logger.info(f"Test patients:  {len(test_names)}")
        return

    # ── Step 4: Copy files to train/ and test/ ───────────────────────────
    logger.info("Step 4: Copying files …")
    logger.info("  Copying train split …")
    train_stats = copy_split(src_dir, dest_dir, patients, train_names, "train", logger)
    logger.info("  Copying test split …")
    test_stats  = copy_split(src_dir, dest_dir, patients, test_names,  "test",  logger)

    # ── Step 5: Write split CSVs ─────────────────────────────────────────
    logger.info("Step 5: Writing split CSVs …")
    write_split_csv(dest_dir, patients, train_names, test_names, metadata_df, logger)

    # ── Step 6: Validate ─────────────────────────────────────────────────
    logger.info("Step 6: Validating …")
    ok = validate_split(src_dir, dest_dir, patients, train_names, test_names, logger)
    if not ok:
        logger.error("Validation FAILED.")
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Train patients : {len(train_names)}")
    logger.info(f"  Test patients  : {len(test_names)}")
    logger.info(f"  Output         : {dest_dir / 'train'}")
    logger.info(f"                   {dest_dir / 'test'}")
    logger.info(f"  Split CSVs     : {dest_dir / 'train_split.csv'}")
    logger.info(f"                   {dest_dir / 'test_split.csv'}")
    logger.info(f"  Log            : {dest_dir / 'preprocess.log'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
