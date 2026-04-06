# Building onnxruntime from source on my Linux fedora 43 setup.
It appears that AMD GPU is only supported via AMDMigraphX which must be pre-installed.

## Linux fedora 43 (latest as of 2/4/2026)

See: https://onnxruntime.ai/docs/build/

## CLONING

```bash
git clone https://github.com/Microsoft/onnxruntime.git
git submodule update --init --recursive
```

Before building install gmock-devel

```bash
sudo dnf install gmock-devel
```

## BUILD

The build.sh command will by default update, build and test.
To see all build options:

```bash
./build.sh --help
```

### BUILDING with migraphx prebuilt installed

```bash
./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --use_migraphx
```







btop shows migraphx as built does not use the GPU, why? My migraphix build must be at fault.
Test execution time 44s.

### BUILDING without migraphx

```bash
./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
```

Gemini suggests

```bash
./build.sh --config Release \
    --update --build --parallel \
    --build_wheel \
    --use_openmp \
    --cmake_extra_defines "CMAKE_CXX_FLAGS=-march=native -O3 -flto" \
    --cmake_extra_defines "CMAKE_C_FLAGS=-march=native -O3 -flto" \
    --skip_tests \
    --compile_no_warning_as_error
```

--use_openmp is not recognized but this works:

```bash
./build.sh --config Release \
    --update --build --parallel \
    --build_wheel \
    --cmake_extra_defines "CMAKE_CXX_FLAGS=-march=native -O3 -flto" \
    --cmake_extra_defines "CMAKE_C_FLAGS=-march=native -O3 -flto" \
    --skip_tests \
    --compile_no_warning_as_error
```

## Test

```bash
cd build/Linux/Release && ./onnxruntime_test_all
```


btop shows no GPU use, as expected.
Test execution time 14508ms.

## Install

Installing to /usr/local do:

```bash
cd build/Linux/Release
make install
```

## Verify Installation

After installation, run:

```bash

pip install build/Linux/Release/dist/onnxruntime-1.25.0-cp314-cp314-linux_x86_64.whl --force-reinstall

#assuming no other onnxruntime is installed

# you must leave the PWD

cd ~
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

also

```bash
onnx_test_runner --help
```

to verify it works.

## To update
```bash
cd ~/Workspace/onnxruntime/onnxruntime
#Next steps not really required, will be done by build script (by default if no --update --build etc. is given)
git pull --recusrse-submodules
rm -Rf build
# build, etc. as above
```
