import os

print(f"using gpu: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
