# Building onnxruntime from source on my Linux Fedora 43 setup

It appears that AMD GPU is only supported via AMD MIGraphX which must be pre-installed.

## Building in Toolbox

For building onnxruntime with AMD ROCm 7.2 on Fedora 43 using AMD supplied rpms in a toolbox, see [build-toolbox-fedora.md](./build-toolbox-fedora.md).

## Linux Fedora 43 (latest as of 2/4/2026)

See: (https://onnxruntime.ai/docs/build/) and also (https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)

## CLONING

`build_onnxruntime.sh` clones the repo automatically if `onnxruntime/` is missing, so manual cloning is optional. To clone manually:

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

*Note: Fedora44 --- builds with migraphx with options below but provider test comes up with errors, they do not seem to be critical.*

*Note: after upgrading Fedora 44 to 45 (GCC 16.2.1, newer ROCm stack) `make test` got much worse --- test 7
`onnxruntime_provider_test` started failing with SEGFAULT plus 3 deterministic RNG assertion failures. Both are fixed by
the patch in `patches/` --- see [Patches](#patches) and [Test Failures](#test-failures) below.*

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

The script clones the repo if missing, force-syncs all submodules to the commits recorded by the root repo (and aborts if any diverge), builds the wheel and shared lib, installs under /usr/local (requires sudo), and pip-installs the wheel.

## Test

```bash
cd build/Linux/Release && make test
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
python test_run_onnx.py onnxruntime/simplified_ssd.onnx
```

## Test Failures

### SEGFAULT in `onnxruntime_provider_test` (Fedora 44 to 45 upgrade)

Symptom: ctest reports test 7 with a signal even though every gtest inside passed and the gtest XML was written:

```
The following tests FAILED:
      7 - onnxruntime_provider_test (SEGFAULT)
```

Root cause: `s_library_migraphx` in `onnxruntime/core/session/provider_bridge_ort.cc` was the only GPU provider still
declared with the default `unload = true`. At process exit `OrtEnv::~OrtEnv()` calls `UnloadSharedProviders()`, which
`dlclose()`s `libonnxruntime_providers_migraphx.so` and its whole ROCm dependency tree (libmigraphx*, libamdhip64,
libhsa-runtime64, MIOpen, rocBLAS, ...). Exit handlers / object teardown that run afterwards then call into code that
has been unmapped and the process dies in `__run_exit_handlers`. This matches the historical pattern already handled
for CUDA/CANN/TensorRT/VitisAI, which are all marked `false /* unload - On Linux if we unload the ... we crash */`
(see issue #5384, PRs #5523/#5553/#24539 upstream).

Diagnosis breadcrumbs: crash RIP is inside an *unmapped* region (gdb: `Cannot access memory`), and
`LD_DEBUG=files,libs` shows `calling fini:` / `destroying link map:` for the migraphx/ROCm libraries immediately
before the crash.

Fix: pass `false` (no unload on Linux) for `s_library_migraphx` --- included in the patch under [Patches](#patches).
This specific instance was not reported upstream at the time of writing (checked Oct 2026); only the general
provider-unload crash class was.

### `onnxruntime_test_all` / `onnxruntime_provider_test` assertion failures

The `onnxruntime_test_all` and `onnxruntime_provider_test` may fail with these tests:

- `QDQTransformerTests.DefaultPath_TagsGeneratedWeightWithStableContentIdentity`
- `Random.MultinomialGoodCase`
- `Random.MultinomialDefaultDType`
- `SamplingTest.Gpt2Sampling_CPU`

**QDQTransformer** (`qdq_matmulnbits_transformer_test.cc:1604`): The test generates a hash-based identity tag for a weight
twice via different code paths and expects both tags to match. The hash value differs between runs due to C++ standard
library hash randomization (ASLR / hash seed randomization). This is a test fragility issue — the two paths should
produce the same identity but the hash function is not stable across executions.

**Multinomial & Sampling (appeared after the Fedora 44 to 45 upgrade):** The tests hardcode expected outputs
generated from `std::default_random_engine` + `std::uniform_real_distribution`. The divergence is **not** in the
engine: Fedora 45 ships GCC 16, and libstdc++ 16 replaced `std::generate_canonical` with a rewrite implementing
P0952R2/LWG 2524 (GCC PR119739). Verified locally: GCC 13/14/15 all reproduce the recorded goldens bit-for-bit,
GCC 16 does not (both g++ and clang++ with libstdc++ 16 headers). `SamplingTest.Gpt2Sampling_CPU` fails for the
same reason --- it goes through `MultinomialComputeShared` with the same RNG.

Fix: define libstdc++'s documented compatibility macro `_GLIBCXX_USE_OLD_GENERATE_CANONICAL` for non-MSVC builds
(global compile definition in `cmake/CMakeLists.txt`), pinning random-op outputs to the historical algorithm on
GCC 16+ instead of regenerating goldens per toolchain. Included in the patch under [Patches](#patches); the three
tests pass afterwards. Note that downstream packagers (Gentoo) were skipping these tests instead
(Gentoo bug 975584, forwarded upstream as onnxruntime issue #28664, which was closed for an unrelated protobuf ABI
problem --- the RNG failures were never fixed upstream).

## Patches

Local fixes that are not (yet) upstream are kept in [`../patches/`](../patches/) and mirrored on the local branch
`dev/fix-migraphx-provider-unload-segfault` in `onnxruntime/`:

| Patch | Fixes | Upstream status (checked Oct 2026) |
|---|---|---|
| `0001-fix-migraphx-provider-unload-segfault-and-gcc16-generate-canonical.patch` | (1) `onnxruntime_provider_test` SEGFAULT at exit: MIGraphX shared provider was `dlclose`d on Linux while CUDA/CANN/TRT/VitisAI are not; (2) `Random.MultinomialGoodCase`, `Random.MultinomialDefaultDType`, `SamplingTest.Gpt2Sampling_CPU` failing on GCC 16 (libstdc++ 16 new `std::generate_canonical`, P0952R2) | (1) not reported upstream; (2) known to packagers (Gentoo skips the tests), not fixed upstream |

To apply after a fresh clone / `git pull` that does not contain the fixes:

```bash
cd onnxruntime
git am ../patches/0001-fix-migraphx-provider-unload-segfault-and-gcc16-generate-canonical.patch
```

`build_onnxruntime.sh` does this automatically before building: it checks whether the patches are already applied
(`git apply --reverse --check`) and, if not, checks out / creates the `dev/local-patches` branch and
`git am --3way`s whatever is missing. The onnxruntime clone itself keeps tracking upstream `main`; the patches and
this README in the outer repo are the source of truth for local fixes.

Re-check the applied state with `git log --oneline -3` (should show the `[ROCM] Do not unload the MIGraphX shared
provider on Linux` and `[cmake] Pin libstdc++ generate_canonical to the pre-GCC 16 algorithm` commits).

## To update (pull latest and rebuild)

```bash
cd ~/Workspace/onnxruntime/onnxruntime
git pull
git submodule update --init --recursive   # optional: build_onnxruntime.sh does this too
./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync
cd build/Linux/Release
make test 2>&1 | tail -20
```
