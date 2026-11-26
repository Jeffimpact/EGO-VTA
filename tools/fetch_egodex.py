"""Fetch selected unmodified EgoDex test episodes directly from Apple's public archive."""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import urllib.request
import zipfile
import zlib
from pathlib import Path

SOURCE = "https://ml-site.cdn-apple.com/datasets/egodex/test.zip"
TASKS = (
    "peel_place_sticker",
    "type_keyboard",
    "pour",
    "basic_pick_place",
    "stack_unstack_cups",
    "screw_unscrew_bottle_cap",
    "add_remove_lid",
)


class RemoteArchive(io.RawIOBase):
    def __init__(self):
        super().__init__()
        with self.request(0, 0) as response:
            self.size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
        self.position = 0

    @staticmethod
    def request(start, stop):
        req = urllib.request.Request(
            SOURCE, headers={"Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity"}
        )
        response = urllib.request.urlopen(req, timeout=30)
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
            f"bytes {start}-{stop}/"
        ):
            response.close()
            raise OSError("The server did not honor a byte range; download stopped.")
        return response

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        origin = {os.SEEK_SET: 0, os.SEEK_CUR: self.position, os.SEEK_END: self.size}[whence]
        if origin + offset < 0:
            raise OSError("Cannot seek before the archive")
        self.position = origin + offset
        return self.position

    def readinto(self, buffer):
        if not buffer or self.position >= self.size:
            return 0
        end = min(self.size - 1, self.position + len(buffer) - 1)
        with self.request(self.position, end) as res:
            data = res.read()
        if len(data) != end - self.position + 1:
            raise OSError("Incomplete archive range")
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def fetch(folder, tasks):
    folder.mkdir(exist_ok=True, parents=True)
    with (
        io.BufferedReader(RemoteArchive(), buffer_size=1024 * 1024) as raw,
        zipfile.ZipFile(raw) as archive,
    ):
        for task in tasks:
            for suffix in ("mp4", "hdf5"):
                info = archive.getinfo(f"test/{task}/0.{suffix}")
                path = folder / f"sample_{task}.{suffix}"
                if path.is_file():
                    if (
                        path.stat().st_size == info.file_size
                        and zlib.crc32(path.read_bytes()) == info.CRC
                    ):
                        print(path, "(cached)")
                        continue
                    raise FileExistsError(f"File differs from the published episode: {path}")
                with tempfile.TemporaryDirectory(dir=folder) as tmp:
                    temp = Path(tmp) / "asset"
                    temp.write_bytes(archive.read(info))  # zipfile checks the member's CRC.
                    temp.rename(path)
                print(path, info.file_size, "bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=["peel_place_sticker", "pour"])
    args = parser.parse_args()
    fetch(args.folder, args.tasks)
