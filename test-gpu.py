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
session = ort.InferenceSession("/home/stoflom/.local/share/darktable/models/denoise-nind/model.onnx", providers=providers)

print(f"Active provider: {session.get_providers()}")
