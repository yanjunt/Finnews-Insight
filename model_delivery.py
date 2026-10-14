"""Download and safely prepare the student fine-tuned model from a GitHub Release asset."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_RELEASE_URL = (
    "https://github.com/yanjunt/Finnews-Insight/releases/download/"
    "finbert-v1.0/finbert-finetuned.zip"
)
REQUIRED_FILES = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
MAX_DOWNLOAD_BYTES = 800 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


def _setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value or default


def _safe_members(zf: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str]]:
    result = []
    total = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/")
        parts = [part for part in name.split("/") if part not in ("", ".")]
        if not parts or any(part == ".." for part in parts) or name.startswith("/"):
            raise ValueError("The model archive contains an unsafe path.")
        if info.file_size > MAX_UNCOMPRESSED_BYTES or total + info.file_size > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("The model archive is larger than the supported extraction limit.")
        result.append((info, "/".join(parts)))
        total += info.file_size
    return result


def _find_model_root(extract_dir: Path) -> Path:
    candidates = [extract_dir]
    candidates.extend(p for p in extract_dir.iterdir() if p.is_dir())
    for candidate in candidates:
        if all((candidate / name).is_file() for name in REQUIRED_FILES):
            return candidate
    raise ValueError("The downloaded archive does not contain a complete fine-tuned model.")


def ensure_model() -> str:
    """Return a local model directory, downloading the configured release once."""
    local = _setting("SENTIMENT_MODEL_PATH")
    if local:
        p = Path(local).expanduser().resolve()
        if all((p / name).is_file() for name in REQUIRED_FILES):
            return str(p)
        raise FileNotFoundError("SENTIMENT_MODEL_PATH does not contain the required model files.")

    url = _setting("SENTIMENT_MODEL_URL", DEFAULT_RELEASE_URL)
    cache_root = Path(_setting("MODEL_CACHE_DIR", "/tmp/finnews-model-cache")).expanduser()
    model_dir = cache_root / "finbert-finetuned"
    if all((model_dir / name).is_file() for name in REQUIRED_FILES):
        return str(model_dir)

    cache_root.mkdir(parents=True, exist_ok=True)
    lock = cache_root / ".download.lock"
    # Streamlit normally runs one process; this lock also prevents concurrent first requests.
    with lock.open("a+") as lock_file:
        try:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        if all((model_dir / name).is_file() for name in REQUIRED_FILES):
            return str(model_dir)

        archive = cache_root / "finbert-finetuned.zip"
        temp_archive = cache_root / "finbert-finetuned.zip.part"
        request = urllib.request.Request(url, headers={"User-Agent": "FinNewsInsight/2.2"})
        with urllib.request.urlopen(request, timeout=180) as response, temp_archive.open("wb") as out:
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_DOWNLOAD_BYTES:
                raise ValueError("The configured model archive exceeds the supported download limit.")
            digest = hashlib.sha256()
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > MAX_DOWNLOAD_BYTES:
                    raise ValueError("The configured model archive exceeds the supported download limit.")
                digest.update(chunk)
                out.write(chunk)
        expected = _setting("SENTIMENT_MODEL_SHA256")
        if expected and digest.hexdigest().lower() != expected.lower():
            temp_archive.unlink(missing_ok=True)
            raise ValueError("The downloaded model archive failed the SHA-256 integrity check.")
        os.replace(temp_archive, archive)

        staging = Path(tempfile.mkdtemp(prefix="finnews-model-", dir=cache_root))
        try:
            with zipfile.ZipFile(archive) as zf:
                members = _safe_members(zf)
                for info, safe_name in members:
                    target = staging / safe_name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
            source_root = _find_model_root(staging)
            if model_dir.exists():
                shutil.rmtree(model_dir)
            shutil.copytree(source_root, model_dir)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return str(model_dir)
