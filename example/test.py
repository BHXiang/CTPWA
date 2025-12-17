import torch
from torch.utils.cpp_extension import load

try:
    ext = load(
        name='mypwa',
        sources=['/home/whitewash/pkgs/ctpwa/src/main.cu'],
        extra_include_paths=[
            '/home/whitewash/pkgs/ctpwa/include',
            '/home/whitewash/miniconda3/envs/ctpwa/include',
            '/home/whitewash/pkgs/root/include',
        ],
        extra_ldflags=[
            '-L/usr/local/cuda-13.0/lib64',
            '-L/home/whitewash/miniconda3/envs/ctpwa/lib',
            '-L/home/whitewash/pkgs/root/lib',
            '-lyaml-cpp', '-lCore', '-lRIO', '-lNet', '-lHist', '-lGraf', '-lGraf3d', '-lGpad',
            '-lTree', '-lMathCore', '-lPhysics', '-lcudart', '-lcublas'
        ],
        extra_cuda_cflags=[
            '-arch=sm_120', 
            '--expt-relaxed-constexpr', '-DTORCH_USE_CUDA_DSA',
            '-Xcompiler', '-fPIC'
        ],
        verbose=True
    )
    print("JIT编译成功!")
except Exception as e:
    print(f"编译错误: {e}")

import mypwa
import torch
import numpy as np
import time

# 初始化分析对象
analysis_instance = mypwa.analysis()

#z = torch.tensor([1+0j, 1+1j], dtype=torch.complex128, device='cuda', requires_grad=True)
z = torch.tensor([1+0j], dtype=torch.complex128, device='cuda', requires_grad=True)

# 使用最佳结果保存权重
analysis_instance.writeWeightFile(z, "weight.root")
#print(f"\n最佳权重已保存到 'weight.root'")

