from __future__ import annotations

import argparse
import sys
import threading
import timeit
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import wwopy

try:
    import pyworld as pw

    exsit_pw = True
except ImportError:
    exsit_pw = False

if sys.version_info >= (3, 11):
    from typing import assert_never

if TYPE_CHECKING:
    import numpy.typing as npt

_TEST_FILE = Path(__file__).parents[1] / "ext/World/test/vaiueo2d.wav"


def test_wave() -> tuple[npt.NDArray[np.double], int]:
    with wave.open(str(_TEST_FILE), "rb") as f:
        f: wave.Wave_read
        nchannels = f.getnchannels()
        sampwidth = f.getsampwidth()
        framerate = f.getframerate()
        buffer = f.readframes(-1)
    if sampwidth == 1:
        dtype = np.dtype("<u1")
    elif sampwidth == 2:
        dtype = np.dtype("<i2")
    else:
        msg = f"sampwidth:{sampwidth} is not support."
        raise Exception(msg)
    data = np.frombuffer(buffer, dtype)
    data = data.astype(np.double)
    if sampwidth == 1:
        data -= 128
        data /= 128
    elif sampwidth == 2:
        data /= 2**15
    else:
        assert_never()
    assert data.max() <= 1
    assert data.min() >= -1
    if nchannels != 1:
        data = data.reshape((-1, nchannels))
    return data, framerate


def print_result(name: str, number, result: list[float]):
    best = min(result)
    print(f"{name:<18}: {number} loops, best of {len(result)}: {best:.4f} per loop")  # noqa: T201


def mt_func(target_func: str, n_thread: int) -> str:
    return f"""
threads = [threading.Thread(target=world.{target_func}, args=args) for i in range({max(n_thread, 2)})]
for i in threads:
    i.start()
for i in threads:
    i.join()
"""


def bench_wwopy(number: int, globals: dict) -> None:
    print("benchmark: wwopy")  # noqa: T201
    data = {"world": wwopy} | globals
    timer = timeit.Timer("world.harvest(wav, fs)", globals=data)
    results = timer.repeat(number=number)
    print_result("hervest", number, results)

    timer = timeit.Timer("world.dio(wav, fs)", globals=data)
    results = timer.repeat(number=number)
    print_result("dio", number, results)

    temporal_positions, f0, frame_period = wwopy.dio(data["wav"], data["fs"])
    timer = timeit.Timer(
        "world.stonemask(wav, fs, tp, f0)",
        globals={"tp": temporal_positions, "f0": f0} | data,
    )
    results = timer.repeat(number=number)
    print_result("stonemask", number, results)

    timer = timeit.Timer(
        "world.cheaptrick(wav, fs, tp, f0)",
        globals={"tp": temporal_positions, "f0": f0} | data,
    )
    results = timer.repeat(number=number)
    print_result("cheaptrick", number, results)

    spectrogram, fft_size = wwopy.cheaptrick(
        data["wav"], data["fs"], temporal_positions, f0
    )
    timer = timeit.Timer(
        "world.d4c(wav, fs, tp, f0, fft_size)",
        globals={"tp": temporal_positions, "f0": f0, "fft_size": fft_size} | data,
    )
    results = timer.repeat(number=number)
    print_result("d4c", number, results)

    ap = wwopy.d4c(data["wav"], data["fs"], temporal_positions, f0, fft_size)
    timer = timeit.Timer(
        "world.synthesis(f0, sp, ap, frame_period, fs)",
        globals={
            "world": wwopy,
            "f0": f0,
            "sp": spectrogram,
            "ap": ap,
            "frame_period": frame_period,
            "fs": data["fs"],
        },
    )
    results = timer.repeat(number=number)
    print_result("synthesis", number, results)


def mt_wwopy_all(n_thread: int) -> str:
    return f"""
def all_action(x, fs):
    temporal_positions, f0, frame_period = world.harvest(x, fs)
    spectrogram, fft_size = world.cheaptrick(x, fs, temporal_positions, f0)
    aperiodicity = world.d4c(x, fs, temporal_positions, f0, fft_size)
    world.synthesis(f0, spectrogram, aperiodicity, frame_period, fs)

threads = [threading.Thread(target=all_action, args=args) for i in range({max(n_thread, 2)})]
for i in threads:
    i.start()
for i in threads:
    i.join()
"""


def bench_wwopy_mt(number: int, n_thread: int, globals: dict) -> None:
    print("benchmark: wwopy multi_threading")  # noqa: T201
    data = {"world": wwopy, "threading": threading}
    args = [globals["wav"], globals["fs"]]
    num = max(number // n_thread, 1)
    timer = timeit.Timer(
        mt_func("harvest", n_thread),
        globals=data | {"args": args},
    )
    results = timer.repeat(number=num)
    print_result("hervest_mt", num, results)

    timer = timeit.Timer(
        mt_func("dio", n_thread),
        globals=data | {"args": args},
    )
    results = timer.repeat(number=num)
    print_result("dio_mt", num, results)

    temporal_positions, f0, frame_period = wwopy.dio(*args)
    stonemask_args = [*args, temporal_positions, f0]
    timer = timeit.Timer(
        mt_func("stonemask", n_thread),
        globals=data | {"args": stonemask_args},
    )
    results = timer.repeat(number=num)
    print_result("stonemask_mt", num, results)

    cheaptrick_args = [*args, temporal_positions, f0]
    timer = timeit.Timer(
        mt_func("cheaptrick", n_thread),
        globals=data | {"args": cheaptrick_args},
    )
    results = timer.repeat(number=num)
    print_result("cheaptrick_mt", num, results)

    spectrogram, fft_size = wwopy.cheaptrick(*args, temporal_positions, f0)
    d4c_args = [*args, temporal_positions, f0, fft_size]
    timer = timeit.Timer(
        mt_func("d4c", n_thread),
        globals=data | {"args": d4c_args},
    )
    results = timer.repeat(number=num)
    print_result("d4c_mt", num, results)

    aperiodicity = wwopy.d4c(*args, temporal_positions, f0, fft_size)
    synthesis_args = [
        f0,
        spectrogram,
        aperiodicity,
        frame_period,
        globals["fs"],
    ]
    timer = timeit.Timer(
        mt_func("synthesis", n_thread),
        globals=data | {"args": synthesis_args},
    )
    results = timer.repeat(number=num)
    print_result("synthesis_mt", num, results)

    timer = timeit.Timer(
        mt_wwopy_all(n_thread),
        globals=data | {"args": args},
    )
    results = timer.repeat(number=num)
    print_result("hervest + cheaptrick + d4c + synthesis", num, results)


def bench_pyworld(number: int, globals: dict) -> None:
    print("benchmark: pyworld")  # noqa: T201
    data = {"world": pw} | globals
    timer = timeit.Timer("world.harvest(wav, fs)", globals=data)
    results = timer.repeat(number=number)
    print_result("pw_hervest", number, results)

    timer = timeit.Timer("world.dio(wav, fs)", globals=data)
    results = timer.repeat(number=number)
    print_result("pw_dio", number, results)

    f0, t = pw.dio(data["wav"], data["fs"])
    dio_data = data | {"f0": f0, "t": t}
    timer = timeit.Timer("world.stonemask(wav, f0, t, fs)", globals=dio_data)
    results = timer.repeat(number=number)
    print_result("pw_stonemask", number, results)

    timer = timeit.Timer("world.cheaptrick(wav, f0, t, fs)", globals=dio_data)
    results = timer.repeat(number=number)
    print_result("pw_cheaptrick", number, results)

    timer = timeit.Timer("world.d4c(wav, f0, t, fs)", globals=dio_data)
    results = timer.repeat(number=number)
    print_result("pw_d4c", number, results)

    sp = pw.cheaptrick(data["wav"], f0, t, data["fs"])  # extract smoothed spectrogram
    ap = pw.d4c(data["wav"], f0, t, data["fs"])
    timer = timeit.Timer(
        "world.synthesize(f0, sp, ap, fs)",
        globals={"world": pw, "f0": f0, "sp": sp, "ap": ap, "fs": data["fs"]},
    )
    results = timer.repeat(number=number)
    print_result("pw_synthesize", number, results)


def mt_pyworld_all(n_thread: int) -> str:
    return f"""
def all_action(x, fs):
    t, f0 = world.harvest(x, fs)
    sp = world.cheaptrick(x, f0, t, fs)
    ap = world.d4c(x, f0, t, fs)
    world.synthesize(f0, sp, ap, fs)

threads = [threading.Thread(target=all_action, args=args) for i in range({max(n_thread, 2)})]
for i in threads:
    i.start()
for i in threads:
    i.join()
"""


def bench_pw_mt(number: int, n_thread: int, globals: dict) -> None:
    print("benchmark: pyworld multi_threading")  # noqa: T201
    data = {"world": pw, "threading": threading}
    args = [globals["wav"], globals["fs"]]
    num = max(number // n_thread, 1)
    timer = timeit.Timer(
        mt_func("harvest", n_thread),
        globals=data | {"args": args},
    )
    results = timer.repeat(number=num)
    print_result("pw_hervest_mt", num, results)

    timer = timeit.Timer(
        mt_func("dio", n_thread),
        globals=data | {"args": args},
    )
    results = timer.repeat(number=num)
    print_result("pw_dio_mt", num, results)

    f0, t = pw.dio(*args)
    args2 = [globals["wav"], f0, t, globals["fs"]]
    timer = timeit.Timer(
        mt_func("stonemask", n_thread),
        globals=data | {"args": args2},
    )
    results = timer.repeat(number=num)
    print_result("pw_stonemask_mt", num, results)

    timer = timeit.Timer(
        mt_func("cheaptrick", n_thread),
        globals=data | {"args": args2},
    )
    results = timer.repeat(number=num)
    print_result("pw_cheaptrick_mt", num, results)

    timer = timeit.Timer(
        mt_func("d4c", n_thread),
        globals=data | {"args": args2},
    )
    results = timer.repeat(number=num)
    print_result("pw_d4c_mt", num, results)

    sp = pw.cheaptrick(*args2)
    ap = pw.d4c(*args2)
    timer = timeit.Timer(
        mt_func("synthesize", n_thread),
        globals=data | {"args": [f0, sp, ap, globals["fs"]]},
    )
    results = timer.repeat(number=num)
    print_result("pw_synthesize_mt", num, results)

    timer = timeit.Timer(
        mt_pyworld_all(n_thread),
        globals=data | {"args": args},
    )
    results = timer.repeat(number=num)
    print_result("pw: hervest + cheaptrick + d4c + synthesis", num, results)


def main(number=64, n_thread=4):
    wav, fs = test_wave()
    data = {"wav": wav, "fs": fs}
    bench_wwopy(number, data)
    bench_wwopy_mt(number, n_thread, data)
    if exsit_pw:
        bench_pyworld(number, data)
        bench_pw_mt(number, n_thread, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", default=64, type=int)
    parser.add_argument("--n-thread", default=4, type=int)
    arg = parser.parse_args()
    main(arg.number, arg.n_thread)
