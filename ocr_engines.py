from __future__ import annotations

import queue as _queue
import threading
from pathlib import Path

import config


_ENGINE_GROUPS = {
    "easyocr": "easyocr",
    "paddleocr": "paddle",
    "tesseract": "tesseract",
}

# ---------------------------------------------------------------------------
# Instance pools: pre-created in ensure_engine() (main thread, sequentially)
# before any worker threads start.  Workers acquire an instance, use it, then
# return it.  This avoids ALL concurrent-import issues with paddle/paddleocr.
# ---------------------------------------------------------------------------
_PADDLE_GPU_POOL: _queue.Queue = _queue.Queue()
_PADDLE_CPU_POOL: _queue.Queue = _queue.Queue()
_EASYOCR_GPU_POOL: _queue.Queue = _queue.Queue()
_EASYOCR_CPU_POOL: _queue.Queue = _queue.Queue()


def extract_text(path: Path, engine_name: str | None = None, use_gpu: bool | None = None) -> str:
    name = (engine_name or config.OCR_ENGINE).strip().lower()
    gpu = config.OCR_USE_GPU if use_gpu is None else use_gpu
    if name == "easyocr":
        return _easyocr(path, gpu)
    if name == "paddleocr":
        return _paddleocr(path, gpu)
    if name == "tesseract":
        return _tesseract(path)
    raise ValueError(f"Unknown OCR_ENGINE={name!r}. Use paddleocr, tesseract, or easyocr.")


def ensure_engine(engine_name: str | None = None) -> None:
    """Pre-initialise all model instances in the calling (main) thread.

    Worker counts come from config.  All heavy imports and model loading happen
    here, sequentially, before any worker thread starts — avoiding concurrent
    paddle/__init__.py execution and circular-import failures.
    """
    name = (engine_name or config.OCR_ENGINE).strip().lower()
    group = _ENGINE_GROUPS.get(name)
    if group is None:
        raise ValueError(f"Unknown OCR_ENGINE={name!r}. Use paddleocr, tesseract, or easyocr.")

    try:
        if name == "paddleocr":
            _init_paddle_pools()
        elif name == "easyocr":
            _init_easyocr_pools()
        else:
            import pytesseract  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"{name} is not installed. Run: uv sync --group {group}"
        ) from exc


# ---------------------------------------------------------------------------
# EasyOCR
# ---------------------------------------------------------------------------

def _easyocr(path: Path, gpu: bool) -> str:
    pool = _EASYOCR_GPU_POOL if gpu else _EASYOCR_CPU_POOL
    reader = pool.get()
    try:
        lines = reader.readtext(str(path), detail=0, paragraph=True)
    finally:
        pool.put(reader)
    return " ".join(lines)


def _init_easyocr_pools() -> None:
    import easyocr

    gpu_w, cpu_w = _worker_counts("easyocr")
    for i in range(gpu_w):
        print(f"  Loading EasyOCR (fa, gpu) [{i+1}/{gpu_w}]...", flush=True)
        _EASYOCR_GPU_POOL.put(easyocr.Reader(["fa"], gpu=True))
    for i in range(cpu_w):
        print(f"  Loading EasyOCR (fa, cpu) [{i+1}/{cpu_w}]...", flush=True)
        _EASYOCR_CPU_POOL.put(easyocr.Reader(["fa"], gpu=False))


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

def _paddleocr(path: Path, gpu: bool) -> str:
    pool = _PADDLE_GPU_POOL if gpu else _PADDLE_CPU_POOL
    ocr = pool.get()
    try:
        if hasattr(ocr, "predict"):
            result = ocr.predict(str(path))
        else:
            result = ocr.ocr(str(path), cls=True)
    finally:
        pool.put(ocr)
    return _paddle_result_to_text(result)


def _init_paddle_pools() -> None:
    use_cuda = _paddle_cuda_available() and config.OCR_USE_GPU
    if config.OCR_USE_GPU and not use_cuda:
        print(
            "  OCR_USE_GPU=true but this PaddlePaddle has no CUDA — using CPU only.\n"
            "  PyPI paddlepaddle is CPU-only; RTX 50-series needs the PaddleX GPU wheel.",
            flush=True,
        )

    gpu_w = _worker_counts("paddleocr")[0] if use_cuda else 0
    cpu_w = _worker_counts("paddleocr")[1]

    _stub_modelscope_if_needed()
    from paddleocr import PaddleOCR as _PaddleOCR  # noqa: F401 — trigger module caching

    for i in range(gpu_w):
        print(f"  Loading PaddleOCR (gpu) [{i+1}/{gpu_w}]...", flush=True)
        _PADDLE_GPU_POOL.put(_build_paddle("gpu"))
    for i in range(cpu_w):
        print(f"  Loading PaddleOCR (cpu) [{i+1}/{cpu_w}]...", flush=True)
        _PADDLE_CPU_POOL.put(_build_paddle("cpu"))


def _worker_counts(engine: str) -> tuple[int, int]:
    """Return (gpu_workers, cpu_workers) from config for the given engine."""
    import os
    cpu_count = os.cpu_count() or 4
    use_gpu = config.OCR_USE_GPU

    if engine == "tesseract":
        return 0, max(1, config.OCR_CPU_WORKERS or config.OCR_WORKERS or min(cpu_count, 4))

    gpu_w = 0
    if use_gpu:
        gpu_w = config.OCR_GPU_WORKERS if config.OCR_GPU_WORKERS > 0 else 1
    cpu_w = config.OCR_CPU_WORKERS or config.OCR_WORKERS or 2
    return gpu_w, max(1, cpu_w)


def _paddle_cuda_available() -> bool:
    try:
        import paddle
        return bool(
            paddle.device.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except Exception:
        return False


def _paddle_lang() -> str:
    raw = (config.OCR_PADDLE_LANG or "fa").strip().lower()
    aliases = {"arabic": "fa", "persian": "fa", "farsi": "fa", "fas": "fa"}
    return aliases.get(raw, raw)


def _stub_modelscope_if_needed() -> None:
    """Block modelscope's `import torch` when paddle DLLs are loaded.

    paddleocr → paddlex → official_models.py → `import modelscope` →
    modelscope.__init__ → torch_utils → `import torch`
    On Windows, torch fails with DLL conflicts after paddle's CUDA DLLs load.
    A minimal stub prevents this; models are already cached locally so modelscope
    is only needed for downloading, not for runtime OCR.
    """
    import sys
    import types

    if "modelscope" in sys.modules:
        return

    def _make(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    ms = _make("modelscope")
    _make("modelscope.hub")
    utils = _make("modelscope.utils")
    fu = _make("modelscope.utils.file_utils")
    fu.get_modelscope_cache_dir = lambda: ""            # type: ignore[attr-defined]
    iu = _make("modelscope.utils.import_utils")
    iu.LazyImportModule = None                          # type: ignore[attr-defined]
    iu.is_name_in_list = lambda *a, **k: False          # type: ignore[attr-defined]
    iu.get_obj_from_cfg = lambda *a, **k: None          # type: ignore[attr-defined]
    lg = _make("modelscope.utils.logger")
    lg.get_logger = lambda: __import__("logging").getLogger("modelscope")  # type: ignore[attr-defined]
    tu = _make("modelscope.utils.torch_utils")
    tu.is_dist = lambda: False                          # type: ignore[attr-defined]
    tu.is_master = lambda: True                         # type: ignore[attr-defined]
    au = _make("modelscope.utils.ast_utils")
    au.INDEX_KEY = "index"                              # type: ignore[attr-defined]
    au.MODULE_KEY = "module"                            # type: ignore[attr-defined]
    au.REQUIREMENT_KEY = "req"                          # type: ignore[attr-defined]
    _make("modelscope.utils.config")
    ms.hub = sys.modules["modelscope.hub"]
    ms.utils = utils


def _build_paddle(device: str):
    from paddleocr import PaddleOCR
    lang = _paddle_lang()
    try:
        init_kwargs: dict = {
            "device": device,
            "use_textline_orientation": True,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        }
        if device == "gpu":
            init_kwargs["lang"] = lang
        else:
            # Paddle 3.3 + oneDNN PIR crashes on Windows CPU; mobile model avoids it.
            init_kwargs.update(
                enable_mkldnn=False,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
            )
        return PaddleOCR(**init_kwargs)
    except TypeError:
        try:
            return PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=(device == "gpu"))
        except TypeError:
            return PaddleOCR(lang=lang)


def _paddle_result_to_text(result) -> str:
    texts: list[str] = []
    if result is None:
        return ""

    if isinstance(result, list) and result and not isinstance(result[0], (list, tuple)):
        for page in result:
            if isinstance(page, dict):
                rec = page.get("rec_texts") or page.get("rec_text") or []
            else:
                rec = getattr(page, "rec_texts", None) or getattr(page, "rec_text", None) or []
            if isinstance(rec, str):
                texts.append(rec)
            else:
                texts.extend(str(t) for t in rec)
        return " ".join(texts)

    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if not page:
            continue
        for line in page:
            if line is None:
                continue
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                rec = line[1]
                if isinstance(rec, (list, tuple)) and rec:
                    texts.append(str(rec[0]))
                else:
                    texts.append(str(rec))
            elif isinstance(line, dict):
                texts.append(str(line.get("text") or line.get("transcription") or ""))
    return " ".join(t for t in texts if t)


# ---------------------------------------------------------------------------
# Tesseract (always CPU, each call is an independent subprocess — thread-safe)
# ---------------------------------------------------------------------------

def _tesseract(path: Path) -> str:
    import pytesseract
    from PIL import Image
    with Image.open(path) as img:
        return pytesseract.image_to_string(img, lang=config.OCR_TESSERACT_LANG)
