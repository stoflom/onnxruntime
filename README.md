# Building onnxruntime from source on my Linux Fedora 43 setup

It appears that AMD GPU is only supported via AMD MIGraphX which must be pre-installed.

## Building in Toolbox

For building onnxruntime with AMD ROCm 7.2 on Fedora 43 using AMD supplied rpms in a toolbox, see [build-toolbox-fedora.md](./build-toolbox-fedora.md).

## Linux Fedora 43 (latest as of 2/4/2026)

See: [https://onnxruntime.ai/docs/build/](https://onnxruntime.ai/docs/build/)

## CLONING

```bash
git clone https://github.com/Microsoft/onnxruntime.git
git submodule update --init --recursive
```

Before building install gmock-devel:

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

*Note: Fedora44 --- builds with migraphx with options below but provider test comes up with 3 errors*

```bash
./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --use_migraphx --migraphx_home "/usr/lib64"
```

### BUILDING without migraphx

*Note: btop shows no GPU use, as expected. Test execution time 14508ms.*

```bash
./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
```

### Optimized Build (Gemini Suggestion)

Note: `--use_openmp` is not recognized, but this works:

```bash
./build.sh --config Release \
    --update --build --parallel \
    --build_wheel \
    --cmake_extra_defines "CMAKE_CXX_FLAGS=-march=native -O3 -flto" \
    --cmake_extra_defines "CMAKE_C_FLAGS=-march=native -O3 -flto" \
    --skip_tests \
    --compile_no_warning_as_error
```

For my tested build script use:

```bash
build_onnxruntime.sh
```

## Test

```bash
cd build/Linux/Release && ./onnxruntime_test_all
```

## Install

Installing to /usr/local:

```bash
cd build/Linux/Release
make install
```

## Verify Installation

After installation, run:

```bash
# Assuming no other onnxruntime is installed
pip install build/Linux/Release/dist/onnxruntime-1.25.0-cp314-cp314-linux_x86_64.whl --force-reinstall

# You must leave the PWD
cd ~
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

Also verify with:

```bash
onnx_test_runner --help
```

Or run python test script from https://github.com/scxiao/ort_test/tree/master/python/run_onnx
```bash
python test_run_onnx.py --default_dim_val 16 /home/stoflom/.local/share/darktable/models/denoise-nind/model.onnx
```

## To update

```bash
cd ~/Workspace/onnxruntime/onnxruntime
# build, etc. as above. The build.sh script does --clean --update --build by default if no flags given.
```
