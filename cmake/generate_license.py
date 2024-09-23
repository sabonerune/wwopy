from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.response import addinfourl

_project_root = Path(__file__).parents[1]


def read_nanobind_license() -> str:
    dist = importlib.metadata.distribution("nanobind")
    text = dist.read_text("licenses/LICENSE")
    if text is None:
        text = fetch_nanobind_license(dist.version)
    return text


# fallback
def fetch_nanobind_license(version: str) -> str:
    url = f"https://github.com/wjakob/nanobind/raw/refs/tags/v{version}/LICENSE"
    with urlopen(url) as r:
        res: addinfourl = r
        return res.read().decode()


def read_license_from_file(filepath: Path) -> str:
    return filepath.read_text("utf-8")


def write_license(dist: Path, licenses: list[tuple[str, str]]):
    delimiter = "\n" + "=" * 80 + "\n\n"
    with dist.open(mode="wt", encoding="utf-8", newline="\n") as f:
        f.write(read_license_from_file(_project_root / "LICENSE.txt") + "\n\n")
        f.write("ThirdPartyLicenses")
        for name, text in licenses:
            f.write(delimiter)
            f.write(f"{name}\n")
            f.write("-" * len(name) + "\n\n")
            f.write(text)


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, required=True)
    parsed_arg = parser.parse_args(args)
    licenses = [
        ("WORLD", read_license_from_file(_project_root / "ext/World/LICENSE.txt")),
        ("nanobind", read_nanobind_license()),
    ]
    write_license(parsed_arg.dist, licenses)


if __name__ == "__main__":
    print("RUN: " + " ".join(sys.argv))  # noqa: T201
    main()
