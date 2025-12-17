#!/usr/bin/env python
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# 获取环境变量中的路径
conda_prefix = os.environ.get('CONDA_PREFIX', '/home/whitewash/miniconda3/envs/ctpwa')
root_dir = os.environ.get('ROOTSYS', '/home/whitewash/pkgs/root')
cuda_dir = os.environ.get('CUDA_HOME', '/usr/local/cuda-13.0')
project_dir = os.path.dirname(os.path.abspath(__file__))

# 定义扩展模块
extension = CUDAExtension(
    name='ctpwa',
    sources=[
        'src/main.cu',
        # 注意：main.cu 包含了其他 .cu 文件，因此不需要单独列出
        # 但如果需要单独编译，可以列出所有源文件
        # 'src/helicity.cu',
        # 'src/Resonance.cu',
        # 'src/AmpGen.cu',
        # 'src/ComputeNLL.cu',
        # 'src/ComputeGrad.cu',
        # 'src/ComputeWeight.cu',
        # 'src/PartialWaveMain.cu',
    ],
    include_dirs=[
        os.path.join(project_dir, 'include'),
        os.path.join(conda_prefix, 'include'),
        os.path.join(root_dir, 'include'),
        os.path.join(cuda_dir, 'include'),
    ],
    library_dirs=[
        os.path.join(cuda_dir, 'lib64'),
        os.path.join(conda_prefix, 'lib'),
        os.path.join(root_dir, 'lib'),
        os.path.join(conda_prefix, 'lib/python3.12/site-packages/torch/lib'),
    ],
    libraries=[
        'yaml-cpp',
        'Core', 'RIO', 'Net', 'Hist', 'Graf', 'Graf3d', 'Gpad',
        'Tree', 'MathCore', 'Physics',
        'cudart', 'cublas',
    ],
    extra_compile_args={
        'cxx': ['-fPIC', '-std=c++17'],
        'nvcc': [
            '-arch=sm_120',  # 根据您的GPU架构调整
            '--expt-relaxed-constexpr',
            '-DTORCH_USE_CUDA_DSA',
            '-Xcompiler', '-fPIC',
            '-std=c++17',
        ]
    }
)

setup(
    name='ctpwa',
    version='0.1.0',
    author='Your Name',
    description='CUDA-accelerated Partial Wave Analysis',
    long_description='A CUDA-accelerated partial wave analysis package for high-energy physics',
    packages=find_packages(exclude=['example', 'config_test']),
    ext_modules=[extension],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.20.0',
        'pyyaml>=5.4.0',  # 用于读取YAML配置文件
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
    ],
)