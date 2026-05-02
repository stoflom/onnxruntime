# UtNet2 Raw Denoiser (RawNIND)

UtNet2 raw denoisers trained on the **Raw Natural Image Noise Dataset (RawNIND)**. This repository contains ONNX models for both Bayer and Linear variants.

## Description

This project provides models for raw and linear-RGB image denoising. It includes:
- **Bayer Variant**: Designed for Bayer-patterned raw sensor data.
- **Linear Variant**: Designed for X-Trans, Foveon, or any already-demosaicked linear-RGB pipeline (Rec.2020 colorspace).

**Author:** Benoit Brummer (UCLouvain)  
**Paper:** [arXiv:2501.08924](https://arxiv.org/abs/2501.08924)  
**License:** GPL-3.0

## Files

- [model_bayer.onnx](https://github.com/stoflom/onnxruntime/raw/main/test_migraphx/model_bayer.onnx): Bayer variant model.
- [model_linear.onnx](https://github.com/stoflom/onnxruntime/raw/main/test_migraphx/model_linear.onnx): Linear variant model.
- `test_denoiser.py`: Test script for evaluating inference performance and correctness.
- `config.json`: Metadata and model attributes.

## Installation

Ensure you have the necessary Python dependencies installed:

```bash
pip install onnxruntime numpy
```

If you intend to use **AMD MIGraphX**, ensure your environment is correctly configured with the appropriate ONNX Runtime build and MIGraphX drivers.

## Usage

The provided `test_denoiser.py` script allows you to test the models on different hardware providers and configurations.

### Basic Commands

**Test the Linear variant on CPU:**
```bash
python test_denoiser.py --provider cpu --variant linear
```

**Test the Bayer variant on MIGraphX:**
```bash
python test_denoiser.py --provider migraphx --variant bayer
```

**Test with a custom tile size:**
```bash
python test_denoiser.py --provider migraphx --variant linear --size 1024
```

### Arguments

| Argument | Choices | Default | Description |
| :--- | :--- | :--- | :--- |
| `--provider` | `cpu`, `migraphx` | `cpu` | Execution provider to use. |
| `--variant` | `linear`, `bayer` | `linear` | Model variant to test. |
| `--size` | `int` | `512` | Input tile size (e.g., 256, 512, 1024). |
| `--force-recompile` | | `False` | Force MIGraphX to recompile by removing existing cached kernels. |

## Performance Optimization (MIGraphX)

The `test_denoiser.py` script automatically applies several optimizations to speed up compilation and subsequent runs:

1.  **Parallel Compilation**: It automatically sets `MIGRAPHX_GPU_COMPILE_PARALLEL` to use all available CPU cores.
2.  **Automatic Model Serialization**: It automatically saves compiled models to the `kernels/` directory and reloads them in subsequent runs to skip compilation entirely.

You can force a fresh compilation at any time using the `--force-recompile` flag.

### Manual Optimization Methods

If you wish to use other optimization methods manually, see below:

#### 1. Using `MIGRAPHX_PROBLEM_CACHE` (Environment Variable)
MIGraphX can store its kernel tuning results and compilation metadata in a persistent JSON file. This is highly effective for speeding up the tuning phase by reusing results from previous runs.

Set the environment variable to a path of your choice:
```bash
export MIGRAPHX_PROBLEM_CACHE="/path/to/your/migraphx_cache.json"
```

#### 2. Custom Model Serialization (Python Options)
If you are writing your own inference code, you can configure the `InferenceSession` as follows:
```python
migraphx_options = {
    "device_id": 0,
    "migraphx_save_model_path": "model_compiled.mxr",
    "migraphx_load_model_path": "model_compiled.mxr",
    "migraphx_fp16_enable": True
}
session = ort.InferenceSession(model_path, providers=[("MIGraphXExecutionProvider", migraphx_options)])
```

### 3. Advanced Tuning (MIGraphX)

For power users looking to squeeze more performance out of the MIGraphX execution provider, several environment variables can control the behavior of the compiler and the kernel selection.

| Environment Variable | Description |
| :--- | :--- |
| `MIGRAPHX_USE_FAST_SOFTMAX` | Set to `1` to enable fast softmax optimization. |
| `MIGRAPHX_ENABLE_LAYERNORM_FUSION` | Enables fusion of LayerNorm operations to reduce memory overhead. |
| `MIGRAPHX_ENABLE_NHWC` | Forces the use of the NHWC data layout. |
| `MIGRAPHX_MLIR_TUNE_EXHAUSTIVE` | If set, enables exhaustive tuning for the MLIR backend. |
| `MIGRAPHX_ENABLE_GEMM_TUNING` | Enables exhaustive tuning for GEMM (General Matrix Multiply) operations. |

*Note: Changing these variables may significantly increase initial compilation time as the compiler explores more optimization paths.*
