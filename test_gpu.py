import torch

device_count = torch.cuda.device_count()
print("Number of CUDA devices:", device_count)

for i in range(device_count):
    device_name = torch.cuda.get_device_name(i)
    print(f"Device {i}: {device_name}")

