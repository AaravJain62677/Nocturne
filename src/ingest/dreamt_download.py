"""Authorized download helper for the DREAMT dataset.

This utility is intentionally defensive: it does not hard-code any dataset URLs,
never stores credentials in Git, and only supports downloads after the user has
completed the relevant access process for an approved provider.

The actual authentication method for a protected dataset is transport-specific and
remains unverified here. The downloader therefore keeps the Authorization header
transport configurable rather than assuming Basic auth is correct.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
from pathlib import Path
from urllib import request

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$")


def _normalize_checksum(checksum: str | None) -> str | None:
    """Validate and normalize a SHA256 checksum string if one is provided."""
    if checksum is None:
        return None
    cleaned = checksum.strip()
    if not cleaned:
        return None
    if not _SHA256_RE.fullmatch(cleaned):
        raise ValueError(
            "Invalid SHA256 checksum format: expected a 64-character hex digest."
        )
    return cleaned.lower()


def _sha256(path: Path) -> str:
    """Return the SHA256 digest for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_checksum(path: Path, expected: str | None) -> bool:
    """Return True when the file matches the expected checksum."""
    normalized = _normalize_checksum(expected)
    if normalized is None:
        return True
    return _sha256(path) == normalized


def _parse_content_range(value: str | None) -> tuple[int, int, int | str] | None:
    """Parse a Content-Range header and return the start, end, and total length."""
    if value is None:
        return None
    match = _CONTENT_RANGE_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Invalid Content-Range header: {value!r}.")
    start = int(match.group(1))
    end = int(match.group(2))
    total = match.group(3)
    if total != "*":
        total = int(total)
    return start, end, total


def _build_auth_headers(
    *,
    username: str | None = None,
    password: str | None = None,
    auth_transport: str = "none",
    auth_value: str | None = None,
) -> dict[str, str]:
    """Build the Authorization header without assuming any single auth scheme."""
    transport = (auth_transport or "none").lower()
    if transport == "none":
        if username is not None or password is not None or auth_value is not None:
            raise ValueError(
                "Authentication values were supplied but auth_transport='none'. "
                "Use 'basic', 'bearer', or 'custom' to set an Authorization header."
            )
        return {}

    if transport == "basic":
        if username is None or password is None:
            raise ValueError(
                "Basic authentication requires both username and password."
            )
        token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        return {"Authorization": f"Basic {token}"}

    if transport == "bearer":
        if auth_value is None or not str(auth_value).strip():
            raise ValueError("Bearer authentication requires auth_value.")
        return {"Authorization": f"Bearer {auth_value}"}

    if transport == "custom":
        if auth_value is None or not str(auth_value).strip():
            raise ValueError(
                "Custom transport requires auth_value containing the full "
                "Authorization header value."
            )
        return {"Authorization": str(auth_value)}

    raise ValueError(
        f"Unsupported auth_transport '{auth_transport}'. Use 'none', 'basic', "
        "'bearer', or 'custom'."
    )


def _stream_to_file(response: request.addinfourl, handle, chunk_size: int) -> None:
    """Read the response in chunks and write it to a file handle."""
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        handle.write(chunk)


def download_file(
    url: str,
    destination: str | Path,
    *,
    checksum: str | None = None,
    username: str | None = None,
    password: str | None = None,
    resume: bool = True,
    chunk_size: int = 1024 * 1024,
    auth_transport: str = "none",
    auth_value: str | None = None,
) -> Path:
    """Download a file to a temporary .part file and move it into place on success.

    A partial download remains in the .part file so a later run can resume. The
    final file is atomically replaced only after successful checksum validation.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    normalized_checksum = _normalize_checksum(checksum)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    part_path = target.with_name(f"{target.name}.part")

    if target.exists() and _validate_checksum(target, normalized_checksum):
        return target
    if part_path.exists() and (
        normalized_checksum is None
        or _validate_checksum(part_path, normalized_checksum)
    ):
        os.replace(part_path, target)
        return target

    auth_headers = _build_auth_headers(
        username=username,
        password=password,
        auth_transport=auth_transport,
        auth_value=auth_value,
    )
    start_byte = part_path.stat().st_size if part_path.exists() else 0

    try:
        if resume and start_byte > 0:
            headers = {**auth_headers, "Range": f"bytes={start_byte}-"}
            request_obj = request.Request(url, headers=headers)
            with request.urlopen(request_obj) as response:
                status = getattr(response, "status", response.getcode())
                if status == 206:
                    content_range = response.headers.get("Content-Range")
                    if content_range is not None:
                        parsed = _parse_content_range(content_range)
                        if parsed is not None and parsed[0] != start_byte:
                            raise ValueError(
                                "Invalid Content-Range for resumed download: "
                                f"server started at byte {parsed[0]}, expected {start_byte}."
                            )
                    with part_path.open("ab") as handle:
                        _stream_to_file(response, handle, chunk_size)
                elif status == 200:
                    part_path.unlink(missing_ok=True)
                    request_obj = request.Request(url, headers=auth_headers)
                    with request.urlopen(request_obj) as fallback:
                        status_fallback = getattr(fallback, "status", fallback.getcode())
                        if status_fallback != 200:
                            raise RuntimeError(
                                "Download resumed from byte zero after server ignored Range, "
                                f"but the response status was {status_fallback}."
                            )
                        with part_path.open("wb") as handle:
                            _stream_to_file(fallback, handle, chunk_size)
                else:
                    raise RuntimeError(
                        f"HTTP error while downloading {url}: unexpected status {status}. "
                        "Verify the URL and configured access policy."
                    )
        else:
            request_obj = request.Request(url, headers=auth_headers)
            with request.urlopen(request_obj) as response:
                status = getattr(response, "status", response.getcode())
                if status not in {200, 206}:
                    raise RuntimeError(
                        f"HTTP error while downloading {url}: unexpected status {status}. "
                        "Verify the URL and configured access policy."
                    )
                with part_path.open("wb") as handle:
                    if status == 206 and part_path.exists() and part_path.stat().st_size > 0:
                        # This is a single-range response that must start at byte zero.
                        content_range = response.headers.get("Content-Range")
                        if content_range is not None:
                            parsed = _parse_content_range(content_range)
                            if parsed is not None and parsed[0] != 0:
                                raise ValueError(
                                    "Invalid Content-Range for a fresh download: "
                                    f"server started at byte {parsed[0]}, expected 0."
                                )
                    _stream_to_file(response, handle, chunk_size)
    except request.HTTPError as exc:
        raise RuntimeError(
            f"HTTP error while downloading {url}: {exc.code} {exc.reason}. "
            "Verify the URL, server, and access policy."
        ) from exc
    except request.URLError as exc:
        raise RuntimeError(
            f"Network failure while downloading {url}: {exc.reason}. "
            "Check connectivity and retry the request."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Timeout while downloading {url}: {exc}. Check connectivity and retry."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Filesystem error while writing {part_path}: {exc}. Check the destination directory."
        ) from exc

    if normalized_checksum is not None and not _validate_checksum(part_path, normalized_checksum):
        raise ValueError(
            f"Downloaded file failed integrity validation for {part_path}."
        )

    os.replace(part_path, target)
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download an official dataset archive after the user has satisfied the "
            "provider's access requirements. The auth transport is configurable "
            "because the precise method for a protected dataset remains unverified "
            "without an authorized provider session."
        )
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Official download URL for the archive or file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the downloaded archive.",
    )
    parser.add_argument(
        "--checksum",
        default=None,
        help="Optional SHA256 checksum for integrity validation.",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="Optional username for authenticated downloads.",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Optional password for authenticated downloads.",
    )
    parser.add_argument(
        "--auth-transport",
        choices=("none", "basic", "bearer", "custom"),
        default="none",
        help=(
            "Authentication transport for the Authorization header. The exact "
            "dataset auth scheme remains unverified here."
        ),
    )
    parser.add_argument(
        "--auth-value",
        default=None,
        help="Optional full Authorization header value for custom auth transports.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable HTTP resume support.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Download chunk size in bytes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        download_file(
            url=args.url,
            destination=args.output,
            checksum=args.checksum,
            username=args.username,
            password=args.password,
            resume=not args.no_resume,
            chunk_size=args.chunk_size,
            auth_transport=args.auth_transport,
            auth_value=args.auth_value,
        )
    except Exception as exc:  # pragma: no cover - CLI boundary path
        parser.exit(status=2, message=f"Download failed: {exc}\n")

    print(f"Downloaded file to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
