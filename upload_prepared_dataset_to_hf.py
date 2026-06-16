import argparse
import inspect
import os
import shutil
from pathlib import Path


DEFAULT_REPO_ID = "RohanRamesh/CAD-SEL"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Upload the prepared CAD-SEL Full layout to a Hugging Face dataset repo."
        )
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to an env file containing HF_TOKEN=...",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help=f"Hugging Face dataset repo id. Defaults to {DEFAULT_REPO_ID}.",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root containing Images/Full and Labels/Full.",
    )
    parser.add_argument(
        "--split-name",
        default="Full",
        help="Prepared split name to upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Upload metadata.xlsx if it can be found.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload prepared CAD-SEL dataset",
        help="Commit message for --upload-mode folder. Large-folder mode creates multiple commits.",
    )
    parser.add_argument(
        "--upload-mode",
        choices=["large", "folder"],
        default="large",
        help="Use resumable upload_large_folder by default; use folder for small test uploads.",
    )
    parser.add_argument(
        "--staging-dir",
        default=".hf_upload_staging",
        help="Local staging directory used by --upload-mode large.",
    )
    parser.add_argument(
        "--rebuild-staging",
        action="store_true",
        help="Delete and recreate the staging directory before uploading.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Worker count for upload_large_folder when supported by your huggingface_hub version.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without creating or uploading anything.",
    )
    return parser.parse_args()


def load_env_file(env_file):
    path = Path(env_file)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Add HF_TOKEN=... to .env or export HF_TOKEN."
        )
    return token


def get_repo_id(args):
    return (
        args.repo_id
        or os.environ.get("HF_REPO_ID")
        or os.environ.get("HUGGINGFACE_REPO_ID")
        or DEFAULT_REPO_ID
    )


def count_files(root, extensions=None):
    if extensions is None:
        return sum(1 for path in root.rglob("*") if path.is_file())
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def find_metadata_file(data_root):
    direct = data_root / "metadata.xlsx"
    if direct.exists():
        return direct

    matches = sorted(data_root.rglob("metadata.xlsx"))
    return matches[0] if matches else None


def validate_prepared_layout(data_root, split_name):
    images_root = data_root / "Images" / split_name
    labels_root = data_root / "Labels" / split_name
    expected_dirs = [
        images_root / "NET-WL",
        images_root / "NET-EUS",
        labels_root / "NET-WL",
        labels_root / "NET-EUS",
    ]

    missing = [path for path in expected_dirs if not path.is_dir()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "Prepared dataset layout is missing required directories:\n"
            f"{formatted}\n"
            "Run download_cad_sel_dataset.py or prepare_cad_sel_dataset.py first."
        )

    return images_root, labels_root


def materialize_file(src, dst):
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return "existing"
        except OSError:
            pass
        dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)

    return "created"


def stage_folder(src_root, dst_root):
    created = 0
    existing = 0

    for src in sorted(src_root.rglob("*")):
        if not src.is_file():
            continue

        dst = dst_root / src.relative_to(src_root)
        status = materialize_file(src, dst)
        if status == "created":
            created += 1
        else:
            existing += 1

    return created, existing


def build_large_upload_staging(args, images_root, labels_root, metadata_file):
    staging_dir = Path(args.staging_dir)
    if args.rebuild_staging and staging_dir.exists():
        shutil.rmtree(staging_dir)

    image_target = staging_dir / "data" / "Images" / args.split_name
    label_target = staging_dir / "data" / "Labels" / args.split_name

    image_created, image_existing = stage_folder(images_root, image_target)
    label_created, label_existing = stage_folder(labels_root, label_target)

    print(
        "Staged images: "
        f"{image_created} created, {image_existing} already present"
    )
    print(
        "Staged labels: "
        f"{label_created} created, {label_existing} already present"
    )

    if metadata_file:
        status = materialize_file(metadata_file, staging_dir / "data" / "metadata.xlsx")
        print(f"Staged metadata: {status}")

    return staging_dir


def call_with_supported_kwargs(func, **kwargs):
    signature = inspect.signature(func)
    supported = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return func(**supported)


def upload_large_folder(api, repo_id, token, staging_dir, num_workers):
    if not hasattr(api, "upload_large_folder"):
        raise RuntimeError(
            "Your huggingface_hub version does not provide upload_large_folder. "
            "Run: pip install -U huggingface_hub hf_xet"
        )

    print(f"Uploading staged folder with upload_large_folder: {staging_dir}")
    call_with_supported_kwargs(
        api.upload_large_folder,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        folder_path=str(staging_dir),
        num_workers=num_workers,
    )


def upload_folder(api, repo_id, token, folder_path, path_in_repo, commit_message):
    print(f"Uploading {folder_path} -> {repo_id}/{path_in_repo}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        folder_path=str(folder_path),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )


def main():
    args = parse_args()
    load_env_file(args.env_file)

    data_root = Path(args.data_root)
    images_root, labels_root = validate_prepared_layout(data_root, args.split_name)
    repo_id = get_repo_id(args)

    image_count = count_files(images_root, IMAGE_EXTENSIONS)
    label_count = count_files(labels_root, {".txt"})

    print(f"Repo: {repo_id}")
    print(f"Images folder: {images_root} ({image_count} image files)")
    print(f"Labels folder: {labels_root} ({label_count} label files)")

    metadata_file = find_metadata_file(data_root) if args.include_metadata else None
    if args.include_metadata:
        if metadata_file:
            print(f"Metadata: {metadata_file}")
        else:
            print("Metadata: not found; continuing without metadata.xlsx")

    if args.dry_run:
        print("Dry run complete. No Hugging Face repo was created or uploaded.")
        return

    from huggingface_hub import HfApi, create_repo

    token = get_token()
    api = HfApi(token=token)

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        private=args.private,
        exist_ok=True,
    )

    if args.upload_mode == "large":
        staging_dir = build_large_upload_staging(
            args=args,
            images_root=images_root,
            labels_root=labels_root,
            metadata_file=metadata_file,
        )
        upload_large_folder(
            api=api,
            repo_id=repo_id,
            token=token,
            staging_dir=staging_dir,
            num_workers=args.num_workers,
        )
    else:
        upload_folder(
            api=api,
            repo_id=repo_id,
            token=token,
            folder_path=images_root,
            path_in_repo=f"data/Images/{args.split_name}",
            commit_message=args.commit_message,
        )
        upload_folder(
            api=api,
            repo_id=repo_id,
            token=token,
            folder_path=labels_root,
            path_in_repo=f"data/Labels/{args.split_name}",
            commit_message=args.commit_message,
        )

        if metadata_file:
            print(f"Uploading {metadata_file} -> {repo_id}/data/metadata.xlsx")
            api.upload_file(
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                path_or_fileobj=str(metadata_file),
                path_in_repo="data/metadata.xlsx",
                commit_message=args.commit_message,
            )

    print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
