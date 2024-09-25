from __future__ import annotations
import argparse
import importlib.metadata
from pathlib import Path

_project_root = Path(__file__).parents[1]


def read_nanobind_license() -> str:
    dist = importlib.metadata.distribution("nanobind")
    version = dist.version
    # NOTE: Use packaging when requirements increase
    filename = "LICENSE" if version == "2.0.0" else "licenses/LICENSE"
    text = dist.read_text(filename)
    if text is None:
        msg = "nanobind license file is not found."
        raise Exception(msg)
    return text


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
    main()
