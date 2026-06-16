# CAD-SEL

CAD-SEL is a dual-modal colonoscopy dataset and baseline object-detection codebase for colorectal subepithelial lesions (SELs). It supports white-light endoscopy (WLE) and endoscopic ultrasound (EUS) images with YOLO-format bounding-box labels.

## Dataset Layout

The downloaded figshare/PDF dataset is organized like this:

```text
data/
  Images/
    Center-1/
      WLE-Set/
        NET_G1/<patient>/*.tiff
        NET_G2/<patient>/*.tiff
        Leiomyoma/<patient>/*.tiff
        Lipoma/<patient>/*.tiff
        NonNeoplasm/<patient>/*.tiff
      EUS-Set/
        ...
    Center-2/
    ...
  Labels/
    Center-1/
      WLE-Set/
        NET_G1/<patient>/*.txt
        ...
  metadata.xlsx
```

The original training code expects a derived layout where modality folders are named `NET-WL` and `NET-EUS`:

```text
data/Images/Full/NET-WL/<category>/<patient>/*.tiff
data/Labels/Full/NET-WL/<category>/<patient>/*.txt
data/Images/Full/NET-EUS/<category>/<patient>/*.tiff
data/Labels/Full/NET-EUS/<category>/<patient>/*.txt
```

In a Kaggle notebook, after cloning this repo, download and prepare everything with:

```bash
python download_cad_sel_dataset.py
```

This downloads the figshare file from `https://figshare.com/articles/dataset/CAD-SEL/29945483?file=62374693`, extracts it under `data/raw`, and creates the training-ready compatibility layout under `data/Images/Full` and `data/Labels/Full`.

If you already downloaded and extracted the dataset manually, create only the compatibility layout:

```bash
python prepare_cad_sel_dataset.py --source data --destination data --split-name Full
```

By default these scripts use `--link-mode auto`, which tries hardlinks first and falls back to copies if needed. Use `--link-mode copy` if you want independent files.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Train a multi-class WLE model:

```bash
python train.py \
    --model_type fasterrcnn_mobilenet_320 \
    --images_dir data/Images/Full/NET-WL \
    --labels_dir data/Labels/Full/NET-WL \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --output_dir output_fasterrcnn_mobilenet_320_WL
```

Train a multi-class EUS model:

```bash
python train.py \
    --model_type fasterrcnn_mobilenet_320 \
    --images_dir data/Images/Full/NET-EUS \
    --labels_dir data/Labels/Full/NET-EUS \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --output_dir output_fasterrcnn_mobilenet_320_EUS
```

For binary NET vs non-NET training, add `--merge_classes`:

```bash
python train.py \
    --model_type fasterrcnn_resnet \
    --images_dir data/Images/Full/NET-WL \
    --labels_dir data/Labels/Full/NET-WL \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --output_dir output_fasterrcnn_resnet_WL_binary \
    --merge_classes
```

## Evaluation

Evaluate a trained model:

```bash
python evaluate.py \
    --model_path output_fasterrcnn_mobilenet_320_WL/best_model_fasterrcnn_mobilenet_320.pth \
    --model_type fasterrcnn_mobilenet_320 \
    --images_dir data/Images/Full/NET-WL \
    --labels_dir data/Labels/Full/NET-WL \
    --output_dir evaluation_results_fasterrcnn_mobilenet_320_Full_WL \
    --conf_threshold 0.5
```

Evaluate a binary model with merged labels:

```bash
python evaluate.py \
    --model_path output_fasterrcnn_resnet_WL_binary/best_model_fasterrcnn_resnet.pth \
    --model_type fasterrcnn_resnet \
    --images_dir data/Images/Full/NET-WL \
    --labels_dir data/Labels/Full/NET-WL \
    --output_dir evaluation_results_fasterrcnn_resnet_Full_WL \
    --conf_threshold 0.5 \
    --merge_classes
```

## Reproducing The Paper

The local dataset matches the CAD-SEL paper totals: 4,912 labelled TIFF images, with 2,817 WLE images and 2,095 EUS images.

This repository now supports the paper's two task types:

- Multi-class detection: class `0` is NET (`NET_G1` and `NET_G2`), class `1` is Leiomyoma, class `2` is Lipoma, and class `3` is NonNeoplasm.
- Binary detection: use `--merge_classes` to train/evaluate NET as class `0` and all non-NET lesions as class `1`.

For exact numerical reproduction of the paper tables, the original train/test split files and training details are still important. The current code performs patient-level random train/validation splitting with seed `42`, which avoids image-level leakage but may not match the authors' unpublished experimental split exactly.

## Upload Prepared Data To Hugging Face

If Kaggle runs out of space while extracting the figshare ZIP, prepare the dataset on a machine with enough disk and upload only the processed layout to Hugging Face:

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=hf_...

python upload_prepared_dataset_to_hf.py \
    --repo-id RohanRamesh/CAD-SEL \
    --include-metadata
```

This uses Hugging Face's resumable `upload_large_folder` path by default. It first builds `.hf_upload_staging/` with hardlinks/copies arranged exactly as they should appear in the dataset repo, then uploads that staging folder. If the upload is interrupted, rerun the same command and it will resume from the local Hugging Face upload cache.

The upload script only sends:

```text
data/Images/Full
data/Labels/Full
data/metadata.xlsx  # optional, with --include-metadata
```

Use `--private` if you want the Hugging Face dataset repo created privately.

For maximum upload throughput, you can also set:

```bash
set HF_XET_HIGH_PERFORMANCE=1   # Windows PowerShell: $env:HF_XET_HIGH_PERFORMANCE="1"
```

In Kaggle, you can then download the prepared Hugging Face dataset instead of the figshare ZIP:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='RohanRamesh/CAD-SEL', repo_type='dataset', local_dir='.', local_dir_use_symlinks=False)"
```

## Supported Models

- `fasterrcnn_resnet`
- `fasterrcnn_resnet50_fpn_v2`
- `fasterrcnn_mobilenet`
- `fasterrcnn_mobilenet_320`
- `fasterrcnn_mobilenet_320_v2`
- `ssd`
- `ssdlite_mobilenet_large`
- `fcos_resnet`
- `retinanet`
- `retinanet_v2`

## Project Structure

```text
CAD-SEL/
  models/model.py
  dataloader.py
  train.py
  evaluate.py
  evaluate_results.py
  download_cad_sel_dataset.py
  prepare_cad_sel_dataset.py
  upload_prepared_dataset_to_hf.py
  run_eval.sh
  README.md
```

## License

This project is licensed under the MIT License. Check the dataset source for the dataset license.
