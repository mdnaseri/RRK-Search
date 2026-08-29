"""Download year OCR newspaper pages from ocr.rrk.ir and copy matches to result/."""

from __future__ import annotations

import argparse
import json
import os
import queue as _queue
import re
import shutil
import threading
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path

import requests
import urllib3

import config
import ocr_engines

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

BASE_HTTPS = "https://ocr.rrk.ir"
BASE_HTTP = "http://ocr.rrk.ir"
DATA_PATH = "/service/rrk/data"
DATA_URLS = (
    f"{BASE_HTTPS}{DATA_PATH}",
    f"{BASE_HTTP}{DATA_PATH}",
)

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,fa;q=0.8",
    "Connection": "keep-alive",
    "Referer": f"{BASE_HTTPS}/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
}


def make_session(cookie: str | None) -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    session.verify = config.VERIFY_SSL
    if cookie:
        session.cookies.set("cookiesession1", cookie)
    return session


def fetch_page(session: requests.Session, page: int, news_text: str = "") -> dict:
    params = {
        "NewsText": news_text,
        "startTime": config.START_DATE,
        "number": "",
        "page": page,
        "pageNumber": "",
        "endTime": config.END_DATE,
        "year": "",
        "month": "",
    }
    last_error = None
    for attempt in range(5):
        for url in DATA_URLS:
            try:
                response = session.get(url, params=params, timeout=90)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_error = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch listing page {page}: {last_error}") from last_error


def image_urls(file_name: str) -> list[str]:
    relative = file_name.lstrip("/")
    return [
        f"{BASE_HTTP}/service/{relative}",
        f"{BASE_HTTPS}/service/{relative}",
    ]


def safe_filename(item: dict) -> str:
    file_name = Path(item["FileName"]).name
    date = item.get("NPDate", "").replace("-", "_")
    number = item.get("Number", "")
    page_no = item.get("PageNumber", "")
    code = item.get("Code", "")
    return f"{date}_n{number}_p{page_no}_{code}_{file_name}"


def download_one(cookie: str | None, item: dict, dest_dir: Path) -> Path | None:
    dest = dest_dir / safe_filename(item)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    session = make_session(cookie)
    last_error = None
    try:
        for url in image_urls(item["FileName"]):
            for attempt in range(4):
                try:
                    response = session.get(url, timeout=90, verify=config.VERIFY_SSL)
                    if response.status_code == 404:
                        break
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    if "json" in content_type or "html" in content_type:
                        break
                    dest.write_bytes(response.content)
                    if dest.stat().st_size == 0:
                        dest.unlink(missing_ok=True)
                        continue
                    return dest
                except requests.RequestException as exc:
                    last_error = exc
                    time.sleep(1.2 * (attempt + 1))
    finally:
        session.close()
    print(f"  download failed: {item.get('FileName')} ({last_error})")
    return None


def download_batch(cookie: str | None, items: list[dict], dest_dir: Path, workers: int) -> dict[str, str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    if not items:
        return mapping
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
        futures = {pool.submit(download_one, cookie, item, dest_dir): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            path = future.result()
            if path:
                mapping[str(item["Code"])] = str(path)
    return mapping


def normalize_fa(text: str) -> str:
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("\u200c", "")
    text = re.sub(r"\s+", "", text)
    return text


NORMALIZED_NEEDLES = [normalize_fa(p) for p in config.SEARCH_PHRASES]


def matching_phrases(text: str) -> list[str]:
    n = normalize_fa(text)
    return [
        phrase
        for phrase, needle in zip(config.SEARCH_PHRASES, NORMALIZED_NEEDLES)
        if needle in n
    ]


def load_ocr_done(ocr_checkpoint: Path) -> set[str]:
    done: set[str] = set()
    if ocr_checkpoint.exists():
        data = json.loads(ocr_checkpoint.read_text(encoding="utf-8"))
        done.update(data.get("processed") or [])
    jsonl_path = ocr_checkpoint.with_suffix(".jsonl")
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    done.add(json.loads(line)["file"])
                else:
                    done.add(line)
    return done


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def empty_checkpoint() -> dict:
    return {
        "list_next_page": 0,
        "list_total": None,
        "list_page_size": None,
        "list_done": False,
        "downloaded": {},
        "search_phrase_index": 0,
        "search_next_page": 0,
        "search_done": False,
        "hit_codes": [],
    }


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return empty_checkpoint()
    data = empty_checkpoint()
    data.update(json.loads(path.read_text(encoding="utf-8")))
    data.setdefault("downloaded", {})
    data.setdefault("hit_codes", [])
    return data


def append_index(index_path: Path, items: list[dict]) -> None:
    with index_path.open("a", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def copy_hit(item: dict, mapping: dict[str, str], result_dir: Path, cookie: str | None) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    code = str(item["Code"])
    src = mapping.get(code)
    if src and Path(src).exists():
        dest = result_dir / Path(src).name
        shutil.copy2(src, dest)
        return
    download_one(cookie, item, result_dir)


def run_local_ocr(
    image_paths: list[Path],
    result_dir: Path,
    ocr_checkpoint: Path,
    engine_name: str,
) -> None:
    done = load_ocr_done(ocr_checkpoint)
    pending = [p for p in image_paths if p.name not in done]
    if not pending:
        print(f"Persian OCR ({engine_name}): all {len(image_paths)} already done.")
        return

    # All model instances are created here (main thread, sequentially).
    ocr_engines.ensure_engine(engine_name)

    gpu_w, cpu_w = ocr_engines._worker_counts(engine_name)
    if not ocr_engines._paddle_cuda_available() or not config.OCR_USE_GPU:
        gpu_w = 0
    parts = ([f"{gpu_w} gpu"] if gpu_w else []) + [f"{cpu_w} cpu"]
    print(
        f"Persian OCR ({engine_name}, {'+'.join(parts)} workers): "
        f"{len(pending)} remaining of {len(image_paths)} images"
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    progress_path = ocr_checkpoint.with_suffix(".jsonl")
    hits_path = result_dir / "ocr_hits.jsonl"

    # Shared queue: both GPU and CPU workers pull from the same image list
    # so whichever is faster automatically processes more images.
    image_q: _queue.Queue[Path] = _queue.Queue()
    for p in pending:
        image_q.put(p)

    result_q: _queue.Queue[tuple[Path, list[str] | None, Exception | None]] = _queue.Queue()
    stop_event = threading.Event()

    def worker(use_gpu: bool) -> None:
        while not stop_event.is_set():
            try:
                path = image_q.get_nowait()
            except _queue.Empty:
                return
            try:
                blob = ocr_engines.extract_text(path, engine_name, use_gpu=use_gpu)
                result_q.put((path, matching_phrases(blob), None))
            except Exception as exc:
                result_q.put((path, None, exc))
            finally:
                image_q.task_done()

    threads: list[threading.Thread] = []
    for _ in range(gpu_w):
        t = threading.Thread(target=worker, args=(True,), daemon=True)
        threads.append(t)
        t.start()
    for _ in range(cpu_w):
        t = threading.Thread(target=worker, args=(False,), daemon=True)
        threads.append(t)
        t.start()

    completed = 0
    fatal: Exception | None = None
    write_lock = threading.Lock()

    while completed < len(pending):
        path, phrases, exc = result_q.get()
        if exc is not None:
            msg = str(exc)
            print(f"  OCR failed {path.name}: {exc}", flush=True)
            if "ConvertPirAttribute2RuntimeAttribute" in msg or "No module named" in msg:
                fatal = exc
                stop_event.set()
                break
            with write_lock:
                append_jsonl(progress_path, {"file": path.name, "engine": engine_name})
            completed += 1
            continue

        with write_lock:
            if phrases:
                dest = result_dir / path.name
                if path.resolve() != dest.resolve():
                    shutil.copy2(path, dest)
                append_jsonl(hits_path, {"file": path.name, "phrases": phrases})
                print(f"  OCR hit: {path.name} ({', '.join(phrases)})", flush=True)
            append_jsonl(progress_path, {"file": path.name, "engine": engine_name})
            completed += 1
            if completed % 5 == 0 or completed == len(pending):
                print(f"  OCR progress: {completed}/{len(pending)}", flush=True)

    stop_event.set()
    for t in threads:
        t.join(timeout=5)

    if fatal:
        raise SystemExit(
            "OCR engine error (same failure would repeat on every image). Fix the engine, then re-run."
        ) from fatal


def collect_images(root: Path, images_dir: Path, state: dict | None = None) -> list[Path]:
    paths: list[Path] = []
    if state:
        paths = [Path(p) for p in state.get("downloaded", {}).values() if Path(p).exists()]
    if not paths and images_dir.exists():
        paths = sorted(
            list(images_dir.glob("*.jpg"))
            + list(images_dir.glob("*.jpeg"))
            + list(images_dir.glob("*.png"))
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OCR newspaper pages and copy phrase matches to result/"
    )
    parser.add_argument("--cookie", default=config.COOKIE or "", help="cookiesession1 value if required")
    parser.add_argument("--images-dir", default=config.IMAGES_DIR, help="folder for all page images")
    parser.add_argument("--result-dir", default=config.RESULT_DIR, help="folder for matching images")
    parser.add_argument("--workers", type=int, default=config.DOWNLOAD_WORKERS, help="parallel downloads per listing page")
    parser.add_argument(
        "--checkpoint",
        default="checkpoint.json",
        help="resume state file",
    )
    parser.add_argument(
        "--ocr-engine",
        default=config.OCR_ENGINE,
        choices=("paddleocr", "tesseract", "easyocr"),
        help="override OCR_ENGINE from .env",
    )
    parser.add_argument(
        "--local-ocr",
        action="store_true",
        help="also run local OCR on downloaded images (slow)",
    )
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="skip download and site search; run Persian OCR on images/",
    )
    return parser.parse_args()


def run_listing(session: requests.Session, cookie: str | None, args: argparse.Namespace, root: Path) -> dict:
    images_dir = root / args.images_dir
    checkpoint_path = root / args.checkpoint
    index_path = root / "index.jsonl"
    state = load_checkpoint(checkpoint_path)

    if state["list_done"]:
        print("Listing already complete; skipping to search.")
        return state

    start_page = int(state["list_next_page"])
    if start_page > 0:
        print(f"Resuming listing from API page {start_page}.")

    page = start_page
    while True:
        payload = fetch_page(session, page, news_text="")
        chunk = list(payload.get("data") or [])
        total = int(payload.get("count") or 0)
        page_size = len(chunk) or int(state.get("list_page_size") or 5)
        total_pages = (total + page_size - 1) // page_size if total else 0

        if state["list_total"] is None:
            print(f"Total records: {total} | page size: {page_size} | pages: {total_pages}")

        if not chunk:
            state["list_done"] = True
            state["list_next_page"] = page
            atomic_write_json(checkpoint_path, state)
            print("Listing finished (empty page).")
            break

        mapping = download_batch(cookie, chunk, images_dir, args.workers)
        state["downloaded"].update(mapping)
        append_index(index_path, chunk)

        next_page = page + 1
        state["list_next_page"] = next_page
        state["list_total"] = total
        state["list_page_size"] = page_size
        if next_page >= total_pages:
            state["list_done"] = True
        atomic_write_json(checkpoint_path, state)

        downloaded_n = len(state["downloaded"])
        print(
            f"  page {page + 1}/{total_pages}: got {len(chunk)} links, "
            f"downloaded {len(mapping)}, total files {downloaded_n}"
        )

        if state["list_done"]:
            print("Listing and downloads complete.")
            break

        page = next_page
        time.sleep(0.15)

    return state


def run_search(session: requests.Session, cookie: str | None, args: argparse.Namespace, root: Path, state: dict) -> dict:
    result_dir = root / args.result_dir
    checkpoint_path = root / args.checkpoint

    if state.get("search_done"):
        print("Search already complete.")
        return state

    hit_codes = set(state.get("hit_codes") or [])
    phrase_index = int(state.get("search_phrase_index") or 0)
    page = int(state.get("search_next_page") or 0)

    print("Searching site OCR text (NewsText)...")
    while phrase_index < len(config.SEARCH_PHRASES):
        phrase = config.SEARCH_PHRASES[phrase_index]
        print(f"  phrase {phrase_index + 1}/{len(config.SEARCH_PHRASES)}: {phrase} (from page {page})")
        while True:
            payload = fetch_page(session, page, news_text=phrase)
            chunk = list(payload.get("data") or [])
            total = int(payload.get("count") or 0)
            page_size = len(chunk) or 5
            total_pages = (total + page_size - 1) // page_size if total else 0

            for item in chunk:
                code = str(item["Code"])
                if code in hit_codes:
                    continue
                hit_codes.add(code)
                copy_hit(item, state["downloaded"], result_dir, cookie)
                print(f"    hit -> {safe_filename(item)}")

            page += 1
            state["search_phrase_index"] = phrase_index
            state["search_next_page"] = page
            state["hit_codes"] = sorted(hit_codes)
            atomic_write_json(checkpoint_path, state)

            if not chunk or page >= total_pages:
                break
            time.sleep(0.15)

        phrase_index += 1
        page = 0
        state["search_phrase_index"] = phrase_index
        state["search_next_page"] = 0
        state["hit_codes"] = sorted(hit_codes)
        atomic_write_json(checkpoint_path, state)

    state["search_done"] = True
    atomic_write_json(checkpoint_path, state)
    print(f"Search complete. Matches: {len(hit_codes)}")
    return state


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    cookie = args.cookie or None
    checkpoint_path = root / args.checkpoint
    ocr_checkpoint = root / f"ocr_checkpoint_{args.ocr_engine}.json"
    result_dir = root / args.result_dir

    try:
        if args.ocr_only:
            state = load_checkpoint(checkpoint_path)
            existing = collect_images(root, root / args.images_dir, state)
            run_local_ocr(existing, result_dir, ocr_checkpoint, args.ocr_engine)
        else:
            session = make_session(cookie)
            state = run_listing(session, cookie, args, root)
            state = run_search(session, cookie, args, root, state)
            if args.local_ocr:
                existing = collect_images(root, root / args.images_dir, state)
                run_local_ocr(existing, result_dir, ocr_checkpoint, args.ocr_engine)
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.")
        raise SystemExit(130)

    hits = list(result_dir.glob("*.jpg")) + list(result_dir.glob("*.jpeg")) + list(result_dir.glob("*.png"))
    print(f"Done. Files in result/: {len(hits)}")


if __name__ == "__main__":
    main()
