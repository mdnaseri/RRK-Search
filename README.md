# RRK OCR search

Downloads Official Gazette page scans from [ocr.rrk.ir](https://ocr.rrk.ir/) for a Jalali date range, then searches Persian phrases in the images. Matching images are copied to `result/`.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```powershell
cd C:\Users\Acer\Desktop\rrk-search
uv sync
Copy-Item .env.example .env
```

Edit `.env` (cookie, dates, phrases, OCR engine).

### Install all OCR engines

Run the setup script **once**. It installs all three engines together and handles the GPU wheel:

```powershell
.\setup.ps1
```

After this you can freely switch `OCR_ENGINE` in `.env` without reinstalling anything.

> **Do not** run `uv sync` on its own to add packages — always use `.\setup.ps1`, which keeps the GPU wheel intact.



## Choose an OCR engine

Set `OCR_ENGINE` in `.env`:


| Value       | When to use                                                                                            |
| ----------- | ------------------------------------------------------------------------------------------------------ |
| `paddleocr` | Best accuracy on noisy scans. Uses GPU automatically if available.                                     |
| `tesseract` | Best for clear printed Persian text. Requires Tesseract + `fas` language pack installed on the system. |
| `easyocr`   | Simple to use; weaker on old newspaper scans.                                                          |




### PaddleOCR

Set `OCR_ENGINE=paddleocr` and `OCR_PADDLE_LANG=fa` (Persian; `arabic` is also accepted).

`OCR_USE_GPU=true` uses the GPU automatically if a CUDA-enabled PaddlePaddle is installed. If CUDA is unavailable it falls back to CPU (oneDNN is disabled on Windows CPU to avoid a Paddle 3.3 crash).

**RTX 50-series on Windows:** the standard PyPI wheel is CPU-only. `setup.ps1` installs PaddleX's [50-series GPU wheel](https://paddlepaddle.github.io/PaddleX/latest/en/installation/paddlepaddle_install.html) (Python 3.12, CUDA 12.9) automatically.

### Tesseract

1. Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) on Windows.
2. During setup, select language data **Persian (**`fas`**)** (and Arabic if available).
3. Set `OCR_ENGINE=tesseract` and `OCR_TESSERACT_LANG=fas+ara`.



### EasyOCR

Set `OCR_ENGINE=easyocr`. No extra system dependencies needed.

**RTX 50-series on Windows:** `setup.ps1` installs `torch+cu128` (the newest CUDA build available for Windows; `cu129`+ are Linux-only). PyTorch 2.11+ with `cu128` includes `sm_120` (Blackwell) kernels. `+cu126` does **not** support RTX 50-series and crashes with `no kernel image is available for execution on the device`.

### Parallel image processing

When `OCR_USE_GPU=true`, GPU and CPU run **simultaneously** — a GPU worker and several CPU workers all pull from the same image queue, so both the GPU and all CPU cores stay busy at once.

| Setting | Default | Effect |
| --- | --- | --- |
| `OCR_GPU_WORKERS` | `1` (when GPU enabled) | Threads that send images to the GPU model |
| `OCR_CPU_WORKERS` | `2` for paddle/easyocr, `4` for tesseract | Threads that process images on CPU in parallel with GPU |

When `OCR_USE_GPU=false` only CPU workers run (no GPU workers).  
Tesseract has no GPU path — it always uses CPU workers only.

## Commands

Run with uv so the project virtualenv is used:

```powershell
# Download pages and search via site API:
uv run python search.py

# Run local OCR only on already-downloaded images in images/:
uv run python search.py --ocr-only

# Download, then also run local OCR:
uv run python search.py --local-ocr

# Override engine for one run:
uv run python search.py --ocr-only --ocr-engine tesseract
```



## `.env` keys


| Key                         | Description                                                     |
| --------------------------- | --------------------------------------------------------------- |
| `RRK_COOKIE`                | `cookiesession1` from the browser (optional)                    |
| `START_DATE` / `END_DATE`   | Jalali date range, e.g. `1360-01-01` … `1360-12-29`             |
| `SEARCH_PHRASES`            | Phrases separated by `|`, e.g. `متالیک|3957411|غفاری`           |
| `OCR_ENGINE`                | `paddleocr` \| `tesseract` \| `easyocr` \| `all`          |
| `OCR_USE_GPU`               | `true` / `false` — PaddleOCR only                               |
| `OCR_PADDLE_LANG`           | `fa` for Persian (alias: `arabic`)                              |
| `OCR_TESSERACT_LANG`        | Tesseract language codes, e.g. `fas+ara`                        |
| `IMAGES_DIR` / `RESULT_DIR` | Directories for downloaded and matched images                   |
| `DOWNLOAD_WORKERS`          | Parallel download threads (default `6`)                         |
| `OCR_GPU_WORKERS`           | GPU inference threads (default `1` when GPU enabled, else `0`)  |
| `OCR_CPU_WORKERS`           | CPU inference threads running alongside GPU (default auto)      |
| `VERIFY_SSL`                | `false` to skip SSL verification (site has an incomplete chain) |




## Output


| Path                            | Contents                                                        |
| ------------------------------- | --------------------------------------------------------------- |
| `images/`                       | Downloaded page JPEGs                                           |
| `result/`                       | Matching page JPEGs — copied immediately when a phrase is found |
| `result/ocr_hits.jsonl`         | One JSON record per match: `{"file": "…", "phrases": […]}`      |
| `checkpoint.json`               | Download/search resume state                                    |
| `ocr_checkpoint_<engine>.jsonl` | OCR resume progress — one record appended per processed image   |
| `index.jsonl`                   | Listing records from the site                                   |


