import torch


def check_gpu_environment():
    """
    检查 PyTorch 的 GPU 环境信息，包括 CUDA 和 cuDNN 的支持、显卡数量等。
    """
    gpu_info = {}

    # 检查是否支持 CUDA
    gpu_info['cuda_available'] = torch.cuda.is_available()

    # 检查是否支持 cuDNN
    gpu_info['cudnn_available'] = torch.backends.cudnn.is_available()

    if gpu_info['cuda_available']:
        # 显卡数量
        gpu_info['num_gpus'] = torch.cuda.device_count()

        # 获取每块显卡的详细信息
        gpu_info['devices'] = []
        for i in range(gpu_info['num_gpus']):
            device_info = {}
            device_info['name'] = torch.cuda.get_device_name(i)
            device_info['total_memory'] = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # 转为 GB
            device_info['capability'] = torch.cuda.get_device_properties(i).major, torch.cuda.get_device_properties(i).minor
            gpu_info['devices'].append(device_info)
    else:
        gpu_info['num_gpus'] = 0

    return gpu_info


def print_gpu_info():
    gpu_info = check_gpu_environment()
    print("GPU Environment Information:")
    
    # CUDA 和 cuDNN 支持情况
    print(f"CUDA Available: {gpu_info['cuda_available']}")
    print(f"cuDNN Available: {gpu_info['cudnn_available']}")
    
    # 显卡信息
    if gpu_info['cuda_available']:
        print(f"Number of GPUs: {gpu_info['num_gpus']}")
        for i, device in enumerate(gpu_info['devices']):
            print(f"GPU {i}: {device['name']}")
            print(f"  Total Memory: {device['total_memory']:.2f} GB")
            print(f"  Compute Capability: {device['capability'][0]}.{device['capability'][1]}")
    else:
        print("No GPUs are available.")


# 测试函数
if __name__ == "__main__":
    print_gpu_info()
