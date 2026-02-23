import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory Capacity: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
