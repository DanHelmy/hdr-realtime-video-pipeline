from __future__ import annotations

import argparse
import dataclasses
import html
import http.cookiejar
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable


ProgressFn = Callable[[str], None] | None

_GOOGLE_DRIVE_USERCONTENT_URL = (
    "https://drive.usercontent.google.com/download"
)
_GOOGLE_DRIVE_UC_URL = "https://drive.google.com/uc"
_DOWNLOAD_TIMEOUT_SECS = 60
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_MANUAL_ASSETS_DRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "1jh8gXBVzqRse-7w_2Dztca1_KVh5eRu1?usp=drive_link"
)


@dataclasses.dataclass(frozen=True)
class RequiredCloneAsset:
    name: str
    relative_path: str
    drive_url: str
    min_size_bytes: int = 1024

    def target_path(self, root_dir: str | os.PathLike[str]) -> Path:
        return Path(root_dir) / self.relative_path


@dataclasses.dataclass(frozen=True)
class AssetEnsureResult:
    asset: RequiredCloneAsset
    path: Path
    status: str
    detail: str = ""


REQUIRED_CLONE_ASSETS: tuple[RequiredCloneAsset, ...] = (
    RequiredCloneAsset(
        name="libmpv-2.dll",
        relative_path=os.path.join("src", "libmpv-2.dll"),
        drive_url=(
            "https://drive.google.com/file/d/"
            "10Xh_lBxetIHIdhcPvwHXauf4ICb_01DP/view?usp=drive_link"
        ),
        min_size_bytes=1024 * 1024,
    ),
    RequiredCloneAsset(
        name="HG_weights.pth",
        relative_path=os.path.join("src", "models", "weights", "HG_weights.pth"),
        drive_url=(
            "https://drive.google.com/file/d/"
            "1dpg31f_XoUGujcWLvE5fkXheDdyXbdFp/view?usp=drive_link"
        ),
        min_size_bytes=1024 * 1024,
    ),
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def manual_assets_drive_url() -> str:
    return _MANUAL_ASSETS_DRIVE_URL


def missing_required_clone_assets(
    root_dir: str | os.PathLike[str] | None = None,
) -> list[RequiredCloneAsset]:
    root = Path(root_dir) if root_dir else project_root()
    missing: list[RequiredCloneAsset] = []
    for asset in REQUIRED_CLONE_ASSETS:
        target = asset.target_path(root)
        if not target.is_file() or target.stat().st_size < asset.min_size_bytes:
            missing.append(asset)
    return missing


def ensure_required_clone_assets(
    root_dir: str | os.PathLike[str] | None = None,
    *,
    force: bool = False,
    progress: ProgressFn = None,
) -> list[AssetEnsureResult]:
    root = Path(root_dir) if root_dir else project_root()
    results: list[AssetEnsureResult] = []
    for asset in REQUIRED_CLONE_ASSETS:
        target = asset.target_path(root)
        if target.is_file() and target.stat().st_size >= asset.min_size_bytes and not force:
            results.append(AssetEnsureResult(asset, target, "present"))
            continue
        try:
            if progress is not None:
                progress(f"Downloading {asset.name} ...")
            _download_google_drive_asset(asset, target)
            results.append(AssetEnsureResult(asset, target, "downloaded"))
        except Exception as exc:
            results.append(AssetEnsureResult(asset, target, "failed", str(exc)))
    return results


def _download_google_drive_asset(asset: RequiredCloneAsset, target: Path) -> None:
    file_id = _extract_google_drive_file_id(asset.drive_url)
    target.parent.mkdir(parents=True, exist_ok=True)

    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookie_jar)
    )
    last_error: Exception | None = None

    candidate_urls = (
        _driveusercontent_download_url(file_id),
        _uc_download_url(file_id),
    )
    for base_url in candidate_urls:
        confirm_token: str | None = None
        for _attempt in range(2):
            url = base_url
            if confirm_token:
                url = _uc_download_url(file_id, confirm=confirm_token)
            try:
                with _open_url(opener, url) as response:
                    head = response.read(64 * 1024)
                    if _response_looks_like_download(response, head):
                        _stream_response_to_file(response, target, head)
                        _validate_download(target, asset)
                        return
                    text = head.decode("utf-8", errors="ignore")
                    confirm_token = _extract_confirm_token(text, cookie_jar)
                    if not confirm_token:
                        raise RuntimeError(
                            f"Google Drive returned an unexpected page for {asset.name}."
                        )
            except Exception as exc:
                last_error = exc
                break

    if last_error is None:
        last_error = RuntimeError(f"Unknown download failure for {asset.name}.")
    raise last_error


def _open_url(opener: urllib.request.OpenerDirector, url: str):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
    )
    return opener.open(request, timeout=_DOWNLOAD_TIMEOUT_SECS)


def _stream_response_to_file(response, target: Path, prefix: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".part",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            if prefix:
                handle.write(prefix)
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _validate_download(target: Path, asset: RequiredCloneAsset) -> None:
    if not target.is_file():
        raise RuntimeError(f"Download did not create {target}.")
    if target.stat().st_size < asset.min_size_bytes:
        raise RuntimeError(f"Downloaded {asset.name} looks incomplete.")
    with target.open("rb") as handle:
        head = handle.read(512).lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError(f"Downloaded {asset.name} is an HTML error page, not the file.")


def _response_looks_like_download(response, head: bytes) -> bool:
    content_disposition = str(response.headers.get("Content-Disposition", "") or "")
    content_type = str(response.headers.get("Content-Type", "") or "").lower()
    if "attachment" in content_disposition.lower():
        return True
    sniff = head.lstrip()[:32].lower()
    if sniff.startswith(b"<!doctype html") or sniff.startswith(b"<html"):
        return False
    return "text/html" not in content_type


def _extract_confirm_token(
    text: str,
    cookie_jar: http.cookiejar.CookieJar,
) -> str | None:
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning") and cookie.value:
            return cookie.value

    patterns = (
        r'name="confirm"\s+value="([^"]+)"',
        r"confirm=([0-9A-Za-z_-]+)",
        r'"confirm":"([^"]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return html.unescape(match.group(1))
    return None


def _driveusercontent_download_url(file_id: str) -> str:
    query = urllib.parse.urlencode(
        {
            "id": file_id,
            "export": "download",
            "confirm": "t",
        }
    )
    return f"{_GOOGLE_DRIVE_USERCONTENT_URL}?{query}"


def _uc_download_url(file_id: str, confirm: str | None = None) -> str:
    query = {
        "export": "download",
        "id": file_id,
    }
    if confirm:
        query["confirm"] = confirm
    return f"{_GOOGLE_DRIVE_UC_URL}?{urllib.parse.urlencode(query)}"


def _extract_google_drive_file_id(url: str) -> str:
    text = str(url or "").strip()
    for pattern in (
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    raise ValueError(f"Unsupported Google Drive URL: {url}")


def _cli_progress(message: str) -> None:
    print(f"[assets] {message}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manage required non-repo assets for HDRTVNet++."
    )
    parser.add_argument(
        "--root",
        default=str(project_root()),
        help="Repo root where assets should be placed.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download any missing required assets into the repo.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download assets even if they already exist.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if args.download_missing:
        results = ensure_required_clone_assets(
            root,
            force=bool(args.force),
            progress=_cli_progress,
        )
        failures = [result for result in results if result.status == "failed"]
        for result in results:
            if result.status == "failed":
                print(
                    f"[assets] FAILED: {result.asset.name} -> {result.path} "
                    f"({result.detail})",
                    file=sys.stderr,
                )
            elif result.status == "downloaded":
                print(f"[assets] OK: {result.asset.name} -> {result.path}")
        return 1 if failures else 0

    missing = missing_required_clone_assets(root)
    if not missing:
        print("[assets] All required assets are present.")
        return 0

    for asset in missing:
        print(f"[assets] MISSING: {asset.name} -> {asset.target_path(root)}")
        print(f"[assets] Source:  {asset.drive_url}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
