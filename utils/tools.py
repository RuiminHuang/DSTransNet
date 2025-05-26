import torch
import random
import numpy as np
import os

def denormalize(image_tensor, mean, std):
    """
    反归一化函数：支持 BCHW 格式的张量。
    
    参数：
        tensor (Tensor): 归一化后的图像数据，形状为 (B, C, H, W)
        mean (Tensor): 归一化时使用的均值，形状为 (C,)
        std (Tensor): 归一化时使用的标准差，形状为 (C,)
    
    返回：
        Tensor: 反归一化后的图像，形状为 (B, C, H, W)
    """
    mean = torch.tensor(mean, dtype=image_tensor.dtype).to(image_tensor.device)
    std = torch.tensor(std, dtype=image_tensor.dtype).to(image_tensor.device)
    # 扩展均值和标准差的维度，使其可以广播到图像数据
    mean = mean[None, :, None, None]  # 扩展到形状 (1, C, 1, 1)
    std = std[None, :, None, None]    # 扩展到形状 (1, C, 1, 1)
    
    # 执行反归一化
    return image_tensor * std + mean


def set_seed(seed=3407):
    # 如果结果不可复现，将会直接报错
    # torch.use_deterministic_algorithms(True)
    random.seed(seed)
    # 为了禁止hash随机化，使得实验可复现
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def init_env(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids




