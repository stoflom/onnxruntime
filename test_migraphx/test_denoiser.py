import onnxruntime as ort
import numpy as np
import time
import argparse
import sys
import os

def run_test():
    parser = argparse.ArgumentParser(description="Test UtNet2 ONNX models with CPU or MIGraphX.")
    parser.add_argument("--provider", choices=["cpu", "migraphx"], default="cpu", help="Execution provider to use.")
    parser.add_argument("--variant", choices=["linear", "bayer"], default="linear", help="Model variant to test.")
    parser.add_argument("--size", type=int, default=512, help="Input size (e.g., 256, 512, 1024).")
    parser.add_argument("--force-recompile", action="store_true", help="Force MIGraphX to recompile the model by ignoring/removing cached kernels.")
    args = parser.parse_args()

    # 1. Map model files based on config.json
    model_map = {
        "bayer": "model_bayer.onnx",
        "linear": "model_linear.onnx"
    }
    model_path = model_map[args.variant]

    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found in the current directory.")
        sys.exit(1)

    # 2. Configure Execution Providers
    if args.provider == "migraphx":
        # Force parallel compilation using all available CPU cores
        nproc = os.cpu_count() or 1
        os.environ["MIGRAPHX_GPU_COMPILE_PARALLEL"] = str(nproc)
        
        kernel_dir = "kernels"
        os.makedirs(kernel_dir, exist_ok=True)
        compiled_model_path = os.path.join(kernel_dir, f"{args.variant}_{args.size}.mxr")

        migraphx_options = {
            'device_id': 0,
            'migraphx_fp16_enable': True,
        }

        if args.force_recompile:
            if os.path.exists(compiled_model_path):
                print(f"Forcing recompile: removing {compiled_model_path}")
                os.remove(compiled_model_path)
        else:
            if os.path.exists(compiled_model_path):
                print(f"Loading compiled model from {compiled_model_path}")
                migraphx_options['migraphx_load_model_path'] = compiled_model_path
            else:
                print(f"No compiled model found. Will save to {compiled_model_path}")
        
        # Always provide save path so it saves if it's a new compilation
        migraphx_options['migraphx_save_model_path'] = compiled_model_path

        providers = [
            ('MIGraphXExecutionProvider', migraphx_options),
            'CPUExecutionProvider'
        ]
        print(f"MIGraphX parallel compilation enabled: {nproc} threads")
    else:
        providers = ['CPUExecutionProvider']

    print(f"--- Testing {args.variant.upper()} variant on {args.provider.upper()} ---")

    # 3. Initialize Session
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        active = session.get_providers()[0]
        print(f"Active Provider: {active}")
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # 4. Prepare Input based on variant attributes
    # Bayer v1 typically uses 4 channels (RGGB); Linear uses 3 (RGB)[cite: 1]
    channels = 4 if args.variant == "bayer" else 3
    shape = (1, channels, args.size, args.size)
    dummy_input = np.random.random(shape).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # 5. Warmup & Inference
    try:
        print(f"Input: {input_name} | Shape: {shape}")
        print("Warmup/Compiling...")
        session.run(None, {input_name: dummy_input})

        print("Running 10 timed iterations...")
        latencies = []
        for _ in range(10):
            start = time.time()
            session.run(None, {input_name: dummy_input})
            latencies.append((time.time() - start) * 1000)

        print(f"\nResults for {args.variant} ({active}):")
        print(f"Average Inference Time: {sum(latencies)/10: .2f} ms")
        
    except Exception as e:
        print(f"Inference Error: {e}")

if __name__ == "__main__":
    run_test()
