from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "VideoRAG-research-dataset-builder/1.0 (educational use)"
QUERIES = (
    'incategory:"Voice of America videos in Mandarin Chinese" filetype:video',
    'incategory:"Videos from China News Service" filetype:video',
    'incategory:"Videos in Standard Mandarin" filetype:video',
    '"中央社影音新聞" filetype:video',
    '"記者會" filetype:video',
    '"记者会" filetype:video',
    '"新聞" filetype:video',
    '"新闻" filetype:video',
)
ALLOWED_LICENSE_MARKERS = (
    "public domain",
    "cc0",
    "cc by",
    "cc-by",
    "creative commons attribution",
)
REJECT_TITLE_MARKERS = (
    "no audio",
    "silent",
    "penalty shoot",
    "football",
    "soccer",
    "music video",
    "concert",
    "娛樂",
    "娱乐",
    "啦啦隊",
    "啦啦队",
    "new student",
    "j-com",
    "cdo ",
    "長樂",
)
JAPANESE_KANA = re.compile(r"[\u3040-\u30ff]")
HAN_TEXT = re.compile(r"[\u3400-\u9fff]")


def fetch_json(params: dict[str, str | int]) -> dict:
    request = Request(f"{API}?{urlencode(params)}", headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=45) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            retry_after = int(exc.headers.get("Retry-After", 5 * (attempt + 1)))
            time.sleep(retry_after)
    raise RuntimeError("Unreachable")


def plain(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    text = str(value.get("value", ""))
    return re.sub(r"<[^>]+>", "", text).strip()


def search_candidates(max_size_mb: float) -> list[dict]:
    candidates: dict[int, dict] = {}
    for query in QUERIES:
        payload = fetch_json(
            {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": 50,
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
            }
        )
        for page in payload.get("query", {}).get("pages", {}).values():
            info_list = page.get("imageinfo") or []
            if not info_list:
                continue
            info = info_list[0]
            mime = str(info.get("mime", ""))
            size = int(info.get("size", 0))
            title = str(page.get("title", ""))
            metadata = info.get("extmetadata", {})
            license_name = plain(metadata.get("LicenseShortName")).lower()
            if not mime.startswith("video/"):
                continue
            if size < 1_000_000 or size > max_size_mb * 1024 * 1024:
                continue
            if not any(marker in license_name for marker in ALLOWED_LICENSE_MARKERS):
                continue
            if any(marker in title.lower() for marker in REJECT_TITLE_MARKERS):
                continue
            if JAPANESE_KANA.search(title) or len(HAN_TEXT.findall(title)) < 2:
                continue
            candidates[int(page["pageid"])] = {
                "page_id": int(page["pageid"]),
                "title": title.removeprefix("File:"),
                "source_page": info.get("descriptionurl"),
                "download_url": info.get("url"),
                "mime": mime,
                "size_bytes": size,
                "license": plain(metadata.get("LicenseShortName")),
                "license_url": plain(metadata.get("LicenseUrl")),
                "artist": plain(metadata.get("Artist")),
                "credit": plain(metadata.get("Credit")),
                "description": plain(metadata.get("ImageDescription")),
                "search_query": query,
            }
        time.sleep(0.2)
    return sorted(candidates.values(), key=lambda item: item["size_bytes"])


def safe_filename(index: int, title: str, url: str) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix not in {".mp4", ".webm", ".ogv", ".ogg"}:
        suffix = ".webm"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(title).stem).strip("._")
    if not stem:
        stem = "mandarin_news"
    return f"{index:03d}_{stem[:100]}{suffix}"


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(7):
        request = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(destination)
            return
        except HTTPError as exc:
            if exc.code != 429 or attempt == 6:
                raise
            retry_after = int(exc.headers.get("Retry-After", 10 * (attempt + 1)))
            time.sleep(retry_after)
    raise RuntimeError("Unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download openly licensed news-like videos from Wikimedia Commons."
    )
    parser.add_argument("--output", type=Path, default=Path("data/raw/open_news"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--max-file-mb", type=float, default=100.0)
    parser.add_argument("--max-total-mb", type=float, default=1536.0)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--use-existing-manifest", action="store_true")
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if args.use_existing_manifest and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = list(existing_manifest["items"])
        total = sum(int(item["size_bytes"]) for item in selected)
    else:
        candidates = search_candidates(args.max_file_mb)
        selected = []
        total = 0
        maximum_total = int(args.max_total_mb * 1024 * 1024)
        for candidate in candidates:
            if len(selected) >= args.count:
                break
            if total + candidate["size_bytes"] > maximum_total:
                continue
            selected.append(candidate)
            total += candidate["size_bytes"]
        if len(selected) < args.count:
            print(f"Warning: only {len(selected)} suitable videos found within limits.")

    downloaded_this_run = 0
    for index, item in enumerate(selected, start=1):
        item.setdefault("local_file", safe_filename(index, item["title"], item["download_url"]))
        print(
            f"{index:02d}. {item['title']} "
            f"({item['size_bytes'] / 1024 / 1024:.1f} MB, {item['license']})"
        )
        if not args.list_only:
            destination = args.output / item["local_file"]
            if not destination.exists() or destination.stat().st_size != item["size_bytes"]:
                if args.batch_size and downloaded_this_run >= args.batch_size:
                    continue
                download(item["download_url"], destination)
                downloaded_this_run += 1
                time.sleep(1.5)

    manifest = {
        "source": "Wikimedia Commons",
        "purpose": "Educational VideoRAG reproduction and evaluation",
        "count": len(selected),
        "total_size_bytes": total,
        "items": selected,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
