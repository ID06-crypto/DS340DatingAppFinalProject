"""Checking, cleaning, and storing photos.

Every upload is decoded with Pillow and re-encoded as a new JPEG. That proves it's a real
image, and it drops EXIF data such as GPS location along with anything hidden in the file.
"""

import io
import os
import re
import secrets
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PHOTOS = 5
MAX_PHOTO_BYTES = 5 * 1024 * 1024
MAX_SIDE = 1600
MAX_PIXELS = 50_000_000  # Refuses "decompression bombs" before decoding them.
ACCEPTED_FORMATS = {"JPEG", "PNG", "WEBP"}
_KEY = re.compile(r"[0-9a-f]{32}\.jpg")


class InvalidPhotoError(ValueError):
    """The upload isn't an image this app accepts."""


def clean_photo(data: bytes) -> bytes:
    """Return the photo as a new JPEG, upright, at most MAX_SIDE pixels on its long side."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            if image.format not in ACCEPTED_FORMATS:
                raise InvalidPhotoError("photos must be JPEG, PNG, or WebP")
            if image.width * image.height > MAX_PIXELS:
                raise InvalidPhotoError("photo dimensions are too large")
            image.load()
            image = ImageOps.exif_transpose(image)  # Keep it upright once EXIF is gone.
    except InvalidPhotoError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError, Warning) as error:
        raise InvalidPhotoError("the file isn't a readable image") from error
    image = image.convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    output = io.BytesIO()
    image.save(output, "JPEG", quality=85, optimize=True)  # No exif= argument, so none is written.
    return output.getvalue()


def new_storage_key() -> str:
    """A random name, never derived from the user or the original filename."""
    return f"{secrets.token_hex(16)}.jpg"


class PhotoStorage(ABC):
    @abstractmethod
    def save(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def load(self, key: str) -> bytes: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class LocalDiskPhotoStorage(PhotoStorage):
    """Stores photos in one directory, which must not be served publicly."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _path(self, key: str) -> Path:
        if not _KEY.fullmatch(key):
            raise ValueError("invalid storage key")
        return self.directory / key

    def save(self, key: str, data: bytes) -> None:
        path = self._path(key)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)

    def load(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
