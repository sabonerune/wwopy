from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

_project_root = Path(__file__).parents[1]


def read_license_from_file(filepath: Path) -> str:
    return filepath.read_text("utf-8")


def write_license(dist: Path, licenses: Iterable[tuple[str, str]]):
    delimiter = "\n" + "=" * 80 + "\n\n"
    with dist.open(mode="wt", encoding="utf-8", newline="\n") as f:
        project_license = (_project_root / "LICENSE.txt").read_text("utf-8")
        f.write(project_license)
        f.write("\n\nThirdPartyLicenses")
        for name, license_path in licenses:
            f.write(delimiter)
            f.write(f"{name}\n")
            f.write("-" * len(name) + "\n\n")
            try:
                license_text = Path(license_path).read_text("utf-8")
            except OSError as e:
                msg = f"Licene Error: {name}"
                raise Exception(msg) from e
            f.write(license_text)


def main(args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="+")
    parser.add_argument("--dist", type=Path, required=True)
    parsed_arg = parser.parse_args(args)
    files: list[str] = parsed_arg.files
    if len(files) % 2 == 1:
        msg = "argment is wrong"
        raise Exception(msg)
    write_license(parsed_arg.dist, zip(files[0::2], files[1::2]))


if __name__ == "__main__":
    main()
