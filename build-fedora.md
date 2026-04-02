# Building onnxruntime from source on my Linux fedora 43 setup.
It appears that AMD GPU is only supported via AMDMigraphX which must be pre-installed.

## Linux fedora 43 (latest as of 2/4/2026)

See: https://onnxruntime.ai/docs/build/

## CLONING

```bash
git clone https://github.com/onnx/onnx.git
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

btop shows no GPU use, as expected.
Test execution time 41s.

## Install

Installing to /usr/local do:

```bash
cd build/Linux/Release
make install
```

## Verify Installation

After installation, run:

```bash
#assuming no other onnxruntime is installed
python -c "import onnx"
```
also

```bash
onnx_test_runner --help
```

to verify it works.
