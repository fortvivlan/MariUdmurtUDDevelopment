"""Download the pinned INEL Nganasan text/annotation archive, without audio."""

import argparse
import hashlib
from pathlib import Path
import urllib.request
from uralic_lm.inel import NGANASAN_URL, NGANASAN_MD5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("rawtexts/sources/nganasan-1.0-lite.zip"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        temporary = args.output.with_suffix(".download")
        print("Downloading INEL Nganasan 1.0 (119.5 MB, no media)...", flush=True)
        urllib.request.urlretrieve(NGANASAN_URL, temporary)
        with temporary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "md5").hexdigest()
        if digest != NGANASAN_MD5:
            raise ValueError("Archive checksum differs from the pinned release; download retained for inspection.")
        temporary.replace(args.output)
    with args.output.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != NGANASAN_MD5:
            raise ValueError("Existing archive does not match INEL Nganasan 1.0.")
    print(args.output)


if __name__ == "__main__":
    main()
