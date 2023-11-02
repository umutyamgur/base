import torch

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    current = torch.cuda.current_device()
    name = torch.cuda.get_device_name(current)
    print(f"cuda is available, with {num_devices} devices, current device index {current} ({name})")
else:
    print(f"cuda is not available")
