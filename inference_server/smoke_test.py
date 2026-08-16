"""Send a small text or image request to an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path


def request(url: str, api_key: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if data is not None:
        headers["Content-Type"] = "application/json"
    query = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(query, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Cannot reach {url}: {error.reason}") from error


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--base-url",
        default=os.environ.get(
            "INFERENCE_BASE_URL", "http://127.0.0.1:8000/v1"
        ),
    )
    result.add_argument("--api-key", default=os.environ.get("INFERENCE_API_KEY", ""))
    result.add_argument("--model", help="served model name; defaults to /v1/models result")
    result.add_argument("--image", action="append", type=Path, default=[])
    result.add_argument("--prompt", default="Describe the visible scene in one concise sentence.")
    result.add_argument("--max-tokens", type=int, default=128)
    return result


def main() -> None:
    arguments = parser().parse_args()
    if len(arguments.image) > 8:
        raise SystemExit("Model profiles accept at most eight images per request")
    missing_images = [str(path) for path in arguments.image if not path.is_file()]
    if missing_images:
        raise SystemExit("Image not found: " + ", ".join(missing_images))
    base_url = arguments.base_url.rstrip("/")
    model_list = request(f"{base_url}/models", arguments.api_key)
    model = arguments.model or model_list["data"][0]["id"]

    if arguments.image:
        content: str | list[dict] = [{"type": "text", "text": arguments.prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": image_data_url(path)}}
            for path in arguments.image
        )
    else:
        content = arguments.prompt
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": arguments.max_tokens,
        "temperature": 0,
    }
    response = request(f"{base_url}/chat/completions", arguments.api_key, payload)
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
