import sys
from importlib.metadata import distribution
from pathlib import PurePosixPath

if __name__ == "__main__":
    dist = distribution("nanobind")
    version = dist.version
    # NOTE: Use packaging when requirements increase
    filename = "LICENSE" if version == "2.0.0" else "licenses/LICENSE"
    target = None
    if dist.files is None:
        msg = 'distribution("nanobind").files is None.'
        raise Exception(msg)
    for file in dist.files:
        if file.match(f"nanobind-*.dist-info/{filename}"):
            target = file
            break
    if target is None:
        msg = "nanobind license file is not found."
        raise Exception(msg)
    sys.stdout.buffer.write(str(PurePosixPath(target.locate())).encode("utf-8"))
