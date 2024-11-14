import sys
from importlib.metadata import distribution

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    dist = distribution("nanobind")
    version = dist.version
    # NOTE: Use packaging when requirements increase
    filename = "LICENSE" if version == "2.0.0" else "licenses/LICENSE"
    target = None
    for file in dist.files:
        if file.match(f"nanobind-*.dist-info/{filename}"):
            target = file
            break
    if target is None:
        msg = "nanobind license file is not found."
        raise Exception(msg)
    print(target.locate())  # noqa: T201
