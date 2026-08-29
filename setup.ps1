# Run ONCE to install all OCR engines + RTX 50-series GPU wheel.
# After this, just change OCR_ENGINE in .env — no reinstalling needed.
# uv caches every download, so re-running this is fast after the first time.

$GPU_WHEEL = "https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp312-cp312-win_amd64.whl"

Write-Host "==> Installing all OCR engines (paddle + easyocr + tesseract)..."
uv sync --group paddle --group easyocr --group tesseract
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Removing CPU paddlepaddle to avoid DLL conflict with GPU wheel..."
uv pip uninstall paddlepaddle 2>&1 | Out-Null

Write-Host "==> Installing RTX 50-series PaddlePaddle GPU wheel (cached after first run)..."
uv pip install --reinstall $GPU_WHEEL
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Installing torch+cu128 for EasyOCR GPU support (RTX 50-series, sm_120)..."
# cu128 is the latest CUDA build available for Windows; cu129+ are Linux-only.
# torch 2.11+ includes sm_120 (Blackwell) kernels so the RTX 50-series works.
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Upgrading nvidia-cudnn-cu12 to match torch's bundled cuDNN (9.19)..."
# paddle wheel installs nvidia-cudnn-cu12==9.9; torch bundles 9.19 DLLs.
# When both are present, Windows loads the wrong version and torch fails to start.
uv pip install "nvidia-cudnn-cu12==9.19.0.56"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Pinning numpy (GPU wheels can bump it; this keeps everything compatible)..."
uv pip install "numpy==2.3.5"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "==> Setup complete. All three engines (paddleocr GPU, easyocr, tesseract) are installed."
Write-Host "    Change OCR_ENGINE in .env, then: uv run python search.py --ocr-only"
