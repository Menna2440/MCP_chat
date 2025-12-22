import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
try:
    import diffusers
    print(f"Diffusers version: {diffusers.__version__}")
except ImportError:
    print("Diffusers not installed")
