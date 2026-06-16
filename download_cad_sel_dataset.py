import argparse
import json
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

from prepare_cad_sel_dataset import prepare_dataset


ARTICLE_ID = "29945483"
FILE_ID = "62374693"
DEFAULT_DOWNLOAD_URL = f"https://ndownloader.figshare.com/files/{FILE_ID}"
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
EXPECTED_IMAGE_COUNT = 4912
EXPECTED_ARCHIVE_SIZE = 11437992919
MIN_ARCHIVE_SIZE = 1024 * 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download CAD-SEL from figshare and prepare it for this repo."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_DOWNLOAD_URL,
        help="Direct dataset archive URL. Defaults to the figshare file from the paper.",
    )
    parser.add_argument(
        "--use-api-url",
        action="store_true",
        help="Resolve the file download URL from the figshare API before downloading.",
    )
    parser.add_argument(
        "--archive-path",
        default=f"data/downloads/CAD-SEL_{FILE_ID}.zip",
        help="Where to store the downloaded archive.",
    )
    parser.add_argument(
        "--extract-dir",
        default="data/raw",
        help="Directory where the archive will be extracted.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where the compatibility layout will be created.",
    )
    parser.add_argument(
        "--split-name",
        default="Full",
        help="Generated split folder name under Images/ and Labels/.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["auto", "hardlink", "copy", "symlink"],
        default="auto",
        help="How to materialize files in the compatibility layout.",
    )
    parser.add_argument(
        "--center-prefix",
        action="store_true",
        help="Prefix patient folders with center names while preparing the compatibility layout.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download again even if the archive already exists.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Remove and recreate the extraction directory before extracting.",
    )
    parser.add_argument(
        "--overwrite-prepared",
        action="store_true",
        help="Replace existing prepared files if they are not already linked to the raw files.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download; use the existing archive or extracted dataset.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Do not extract; use the existing extracted dataset.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the final expected-image-count check.",
    )
    return parser.parse_args()


def request_url(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
            ),
            "Accept": "*/*",
        },
    )
    return urllib.request.urlopen(request)


def resolve_download_url():
    try:
        with request_url(FIGSHARE_API_URL) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"Could not resolve figshare API URL, using default URL: {exc}")
        return DEFAULT_DOWNLOAD_URL

    for file_info in payload.get("files", []):
        if str(file_info.get("id")) == FILE_ID:
            return file_info.get("download_url") or DEFAULT_DOWNLOAD_URL

    print(f"File id {FILE_ID} was not present in figshare API response; using default URL.")
    return DEFAULT_DOWNLOAD_URL


def download_file(url, archive_path, force=False):
    archive_path = Path(archive_path)
    if archive_path.exists() and not force:
        print(f"Archive already exists: {archive_path}")
        validate_archive_download(archive_path)
        return archive_path

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".part")

    print(f"Downloading: {url}")
    print(f"Target: {archive_path}")

    with request_url(url) as response, tmp_path.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        next_report = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break

            output.write(chunk)
            downloaded += len(chunk)

            if downloaded >= next_report:
                if total:
                    pct = downloaded / total * 100
                    print(
                        f"  {downloaded / 1024 / 1024:.1f} MiB / "
                        f"{total / 1024 / 1024:.1f} MiB ({pct:.1f}%)"
                    )
                else:
                    print(f"  {downloaded / 1024 / 1024:.1f} MiB")
                next_report = downloaded + 256 * 1024 * 1024

    tmp_path.replace(archive_path)
    validate_archive_download(archive_path)
    print("Download complete.")
    return archive_path


def validate_archive_download(archive_path):
    size = archive_path.stat().st_size
    print(f"Downloaded size: {size / 1024 / 1024 / 1024:.2f} GiB")

    if size < MIN_ARCHIVE_SIZE:
        magic = read_magic_bytes(archive_path)
        raise RuntimeError(
            f"Downloaded file is too small to be the CAD-SEL archive: {archive_path} "
            f"({size} bytes). First bytes: {magic!r}. "
            "Try rerunning with --force-download --use-api-url."
        )

    if EXPECTED_ARCHIVE_SIZE and abs(size - EXPECTED_ARCHIVE_SIZE) > 1024 * 1024:
        print(
            "Warning: downloaded size differs from figshare metadata. "
            f"Expected about {EXPECTED_ARCHIVE_SIZE} bytes, got {size} bytes."
        )


def safe_extract_zip(archive_path, extract_dir):
    extract_root = extract_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (extract_dir / member.filename).resolve()
            if not is_relative_to(target, extract_root):
                raise ValueError(f"Refusing unsafe zip path: {member.filename}")
        archive.extractall(extract_dir)


def safe_extract_tar(archive_path, extract_dir):
    extract_root = extract_dir.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (extract_dir / member.name).resolve()
            if not is_relative_to(target, extract_root):
                raise ValueError(f"Refusing unsafe tar path: {member.name}")
        archive.extractall(extract_dir)


def is_relative_to(path, parent):
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def extract_archive(archive_path, extract_dir, force=False):
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)

    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)

    existing_root = find_dataset_root(extract_dir)
    if existing_root and not force:
        print(f"Extracted dataset already exists: {existing_root}")
        return existing_root

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} to {extract_dir}")

    if zipfile.is_zipfile(archive_path):
        safe_extract_zip(archive_path, extract_dir)
    elif tarfile.is_tarfile(archive_path):
        safe_extract_tar(archive_path, extract_dir)
    else:
        extract_with_7z(archive_path, extract_dir)

    dataset_root = find_dataset_root(extract_dir)
    if not dataset_root:
        raise FileNotFoundError(
            f"Could not find extracted Images/ and Labels/ folders under {extract_dir}"
        )

    print(f"Extracted dataset root: {dataset_root}")
    return dataset_root


def extract_with_7z(archive_path, extract_dir):
    seven_zip = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if not seven_zip:
        magic = read_magic_bytes(archive_path)
        raise ValueError(
            f"Unsupported archive format: {archive_path}. "
            f"First bytes: {magic!r}. Install p7zip/7z or pass a ZIP/TAR archive."
        )

    print("Archive is not ZIP/TAR according to Python; trying 7z extractor.")
    command = [
        seven_zip,
        "x",
        "-y",
        f"-o{extract_dir}",
        str(archive_path),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        magic = read_magic_bytes(archive_path)
        raise ValueError(
            f"7z could not extract {archive_path}. "
            f"First bytes: {magic!r}."
        )


def read_magic_bytes(path, size=64):
    with Path(path).open("rb") as handle:
        return handle.read(size)


def find_dataset_root(root):
    root = Path(root)
    if not root.exists():
        return None

    candidates = [root]
    candidates.extend(path for path in root.rglob("*") if path.is_dir())

    for candidate in candidates:
        if (candidate / "Images").is_dir() and (candidate / "Labels").is_dir():
            return candidate

    return None


def verify_prepared_dataset(output_dir, split_name):
    image_root = Path(output_dir) / "Images" / split_name
    counts = {}
    total = 0

    for modality in ["NET-WL", "NET-EUS"]:
        modality_root = image_root / modality
        count = sum(
            1
            for path in modality_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        )
        counts[modality] = count
        total += count

    print("Prepared image counts:")
    for modality, count in counts.items():
        print(f"  {modality}: {count}")
    print(f"  total: {total}")

    if total != EXPECTED_IMAGE_COUNT:
        raise SystemExit(
            f"Expected {EXPECTED_IMAGE_COUNT} prepared images, found {total}. "
            "Use --no-verify to bypass this check."
        )


def main():
    args = parse_args()

    url = resolve_download_url() if args.use_api_url else args.url
    archive_path = Path(args.archive_path)
    extract_dir = Path(args.extract_dir)

    if args.skip_download:
        print("Skipping download.")
    else:
        download_file(url, archive_path, force=args.force_download)

    if args.skip_extract:
        print("Skipping extraction.")
        dataset_root = find_dataset_root(extract_dir)
        if not dataset_root:
            raise FileNotFoundError(
                f"Could not find Images/ and Labels/ under {extract_dir}"
            )
    else:
        dataset_root = extract_archive(
            archive_path,
            extract_dir,
            force=args.force_extract,
        )

    prepare_args = SimpleNamespace(
        source=str(dataset_root),
        destination=args.output_dir,
        split_name=args.split_name,
        link_mode=args.link_mode,
        center_prefix=args.center_prefix,
        overwrite=args.overwrite_prepared,
        dry_run=False,
    )
    stats = prepare_dataset(prepare_args)

    print(f"Created compatibility split: {args.split_name}")
    print(f"Image target: {Path(args.output_dir) / 'Images' / args.split_name}")
    print(f"Label target: {Path(args.output_dir) / 'Labels' / args.split_name}")
    print(f"Mode: {args.link_mode}")
    print(f"Images created: {stats['images_created']}")
    print(f"Images already present: {stats['images_existing']}")
    print(f"Labels created: {stats['labels_created']}")
    print(f"Labels already present: {stats['labels_existing']}")
    print(f"Missing labels: {stats['missing_labels']}")

    if stats["missing_labels"]:
        raise SystemExit("Some images did not have matching YOLO label files.")

    if not args.no_verify:
        verify_prepared_dataset(args.output_dir, args.split_name)


if __name__ == "__main__":
    main()
