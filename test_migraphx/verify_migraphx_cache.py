
import os
import shutil
import tempfile
import subprocess
import numpy as np
import onnxruntime as ort
import time

# Paths to the test models
MODEL_PATH = "/home/stoflom/Workspace/onnxruntime/test_migraphx/model_bayer.onnx"

# List of environment variables to probe
ENV_VARS_TO_TEST = [
    "MIGRAPHX_MODEL_CACHE_DIR",
    "ORT_MIGRAPHX_MODEL_CACHE_DIR",
    "ORT_MIGRAPHX_MODEL_CACHE_PATH",
    "MIGRAPHX_MODEL_CACHE_PATH",
    "MIGRAPHX_MODEL_DIR",
    "MIGRAPHX_CACHE_DIR",
    "ORT_MIGRAPHX_CACHE_DIR"
]

def test_env_var(var_name):
    print(f"[*] Testing {var_name}...")

    # Create a temporary directory for the cache
    cache_dir = tempfile.mkdtemp(prefix=f"test_{var_name}_")
    print(f"    Target cache directory: {cache_dir}")

    try:
        # Clear existing env vars if they might interfere
        # We want to be sure we are only testing one at a time
        for v in ENV_VARS_TO_TEST:
            if v in os.environ:
                del os.environ[v]

        # Set the environment variable we are testing
        os.environ[var_name] = cache_dir

        # Also try setting it via the session options just in case
        # as some versions of ORT might prefer one over the other
        # but we are primarily testing ENV.

        # Initialize session
        # We use 'MIGraphXExecutionProvider'
        # Note: if MIGraphX is not available, this will fail.
        try:
            # We also want to pass it in session options to see if it works
            # when the EP is initialized.
            # However, the goal is to see if the ENV var is picked up.
            session = ort.InferenceSession(MODEL_PATH, providers=['MIGraphXExecutionProvider', 'CPUExecutionProvider'])
        except Exception as e:
            print(f"    [!] Failed to initialize session: {e}")
            return False

        # Prepare dummy input
        # Based on test_denoiser.py, bayer is 4 channels
        shape = (1, 4, 128, 128 )
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.random(shape).astype(np.float32)

        print("    Running inference (compilation stage)...")
        start_time = time.time()
        session.run(None, {input_name: dummy_input})
        duration = time.time() - start_time
        print(f"    Inference took {duration:.2f}s")

        # Check for .mxr files in the cache directory
        files = os.listdir(cache_dir)
        mxr_files = [f for f in files if f.endswith('.mxr')]

        if mxr_files:
            print(f"    [SUCCESS] Found {len(mxr_files)} .mxr files: {mxr_files}")
            return True
        else:
            print("    [FAILURE] No .mxr files found in cache directory.")
            return False

    except Exception as e:
        print(f"    [!] Error during test: {e}")
        return False
    finally:
        # Cleanup
        if var_name in os.environ:
            del os.environ[var_name]
        # shutil.rmtree(cache_dir) # Keep it for a bit if you want to inspect, but for automation we cleanup

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    print("Starting MIGraphX Cache Environment Variable Probe")
    print("====================================================")

    results = []
    for var in ENV_VARS_TO_TEST:
        success = test_env_var(var)
        results.append((var, success))
        print("-" * 50)

    print("\nSummary of results:")
    print("====================================================")
    for var, success in results:
        status = "OK" if success else "FAILED"
        print(f"{var:<30} : {status}")

if __name__ == "__main__":
    main()
