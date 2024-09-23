# wwopy: WORLD Wrapper of Python.

A Python wrapper for [WORLD](https://github.com/mmorise/World).

> [!NOTE]
> There is a similar package [PyWORLD](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder).  
> We usually recommend using that package.

## Installation

Install Python 3.8+ and C++ 17 compiler.

It is not currently released on pypi.  
```shell
python -m pip install https://github.com/sabonerune/wwopy
```

## API

See docstring and test.

For argument specifications, see the [WORLD](https://github.com/mmorise/World) repository.

## Development

This project uses [scikit-build-core](https://github.com/scikit-build/scikit-build-core) and [nanobind](https://github.com/wjakob/nanobind).  
Check their documentation for more information.

### Editable installs

```Shell
git submodule update --init --recursive
python -m pip install --upgrade pip
python -m pip install --requirement ./requirements.txt
python -m pip install --no-build-isolation \
    --config-settings=editable.rebuild=true \
    --config-settings=build-dir=build \
    --editable .[dev,test]
```

If you need detailed logs, add the following options:  
```Shell
python -m pip install --no-build-isolation \
    --config-settings=editable.rebuild=true \
    --config-settings=build-dir=build \
    --config-settings=build.verbose=true \
    --config-settings=logging.level=INFO \
    --verbose --editable .[dev,test]
```

### Test

```Shell
python -m pytest
```

### Format

``` Shell
python -m ruff check --fix
clang-format -i ./src/wwopy_ext.cpp
cmake-format --in-place ./CMakeLists.txt
```
