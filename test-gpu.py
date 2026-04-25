import onnx
import onnxruntime as ort

# Explicitly request the MIGraphX provider
providers = [
    ('MIGraphXExecutionProvider', {
        'device_id': 0,
        'migraphx_fp16_enable': True,  # Highly recommended for AMD iGPU/dGPU performance
    }),
    'CPUExecutionProvider'
]

# Load your model (e.g., one used in your agentic workflow)
model_path = "/home/stoflom/.local/share/darktable/models/denoise-nind/model.onnx"
session = ort.InferenceSession(model_path, providers=providers)

print(f"Active provider: {session.get_providers()}")

print(f"The model description is: {session.get_modelmeta().description}")

# Check the model
try:
    onnx.checker.check_model(model_path)
except onnx.checker.ValidationError as e:
    print(f"The model is invalid: {e}")
else:
    print("The model is valid!")
    # Get input details
    model_inputs = session.get_inputs()
    for input in model_inputs:
        print(f"Name: {input.name}")
        print(f"Shape: {input.shape}") # Look for -1 (dynamic) or fixed numbers
        print(f"Type: {input.type}\n")
