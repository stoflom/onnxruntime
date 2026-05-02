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

## Performance Optimization (MIGraphX)

If you are using the MIGraphX execution provider, the initial compilation can be slow. You can optimize subsequent runs using the following methods:

### 1. Parallel Compilation
Speed up the initial graph compilation by using multiple CPU cores:
```bash
export MIGRAPHX_GPU_COMPILE_PARALLEL=$(nproc)
```

### 2. Persistent Compilation Cache

#### Method A: Using `MIGRAPHX_PROBLEM_CACHE` (Environment Variable)
MIGraphX can store its kernel tuning results and compilation metadata in a persistent JSON file. This is highly effective for speeding up the tuning phase by reusing results from previous runs.

Set the environment variable to a path of your choice:
```bash
export MIGRAPHX_PROBLEM_CACHE="/path/to/your/migraphx_cache.json"
```

#### Method B: Model Serialization (Python Options)
This method saves the entire optimized MIGraphX graph as a serialized binary file. This is often the fastest way to skip compilation entirely for a specific model.

In your Python code, configure the `InferenceSession` as follows:
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
